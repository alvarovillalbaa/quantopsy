#!/usr/bin/env python3
"""Independent bounded public-GGUF descriptor observer; no Quantopsy imports."""

import argparse
import base64
import codecs
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import sys
import time


SPEC = "https://github.com/ggml-org/ggml/blob/facdc44a67da17df4433adcc468b424f48e9ded2/docs/gguf.md"
FIXTURES = {
    "stories260k-f32": {
        "bytes": 1185376,
        "sha256": "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d",
        "revision": "479896ec924af6d40fd419ab8f4d1eb2101de00d",
        "url": "https://huggingface.co/ggml-org/test-model-stories260K/resolve/479896ec924af6d40fd419ab8f4d1eb2101de00d/stories260K-f32.gguf",
    },
    "stories15m-reference": {
        "bytes": 98357920,
        "sha256": "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a",
        "revision": "4cd12190a40543c61bad59a1b8d982dfc48de020",
        "url": "https://huggingface.co/ggml-org/models-moved/resolve/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M.gguf",
    },
    "stories15m-q4_0": {
        "bytes": 19077344,
        "sha256": "6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04",
        "revision": "4cd12190a40543c61bad59a1b8d982dfc48de020",
        "url": "https://huggingface.co/ggml-org/models-moved/resolve/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M-q4_0.gguf",
    },
}
LIMITS = {
    "workSeconds": 90, "stdoutSeconds": 2, "reportBytes": 2 * 1024**2,
    "metadataEncodedBytes": 32 * 1024**2, "metadataEntries": 4096,
    "arrayItems": 262144, "metadataValues": 1048576, "stringBytes": 1024**2,
    "keyBytes": 1024, "tensorNameBytes": 64, "tensors": 8192, "dimensions": 4,
    "descriptorBytes": 256 * 1024, "retainedScalarStringBytes": 2048,
    "sourceBytes": 128 * 1024, "alignment": 4096, "streamBytes": 1024**2,
}
SCALAR_KEYS = {
    "general.architecture", "general.name", "general.alignment", "general.file_type",
    "general.quantization_version", "llama.context_length", "llama.embedding_length",
    "llama.feed_forward_length", "llama.block_count", "llama.attention.head_count",
    "llama.attention.head_count_kv", "llama.rope.dimension_count",
    "llama.attention.layer_norm_rms_epsilon",
}
NUMERIC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
START = time.monotonic()


class Refusal(Exception):
    """Finite, non-payload diagnostic."""


def require(condition, code):
    if not condition:
        raise Refusal(code)


def clock_check():
    require(time.monotonic() - START < LIMITS["workSeconds"], "inventory_deadline")


def identity(value):
    return {"device": str(value.st_dev), "inode": str(value.st_ino), "uid": value.st_uid,
            "mode": format(stat.S_IMODE(value.st_mode), "04o"), "bytes": value.st_size,
            "mtimeNs": str(value.st_mtime_ns), "ctimeNs": str(value.st_ctime_ns)}


def fingerprint(fd, expected):
    digest, offset = hashlib.sha256(), 0
    while offset < expected:
        clock_check()
        chunk = os.pread(fd, min(LIMITS["streamBytes"], expected - offset), offset)
        require(bool(chunk), "input_truncated_during_hash")
        digest.update(chunk)
        offset += len(chunk)
    require(os.pread(fd, 1, offset) == b"", "input_grew_during_hash")
    return digest.hexdigest()


def owned_path(root, file):
    require(file.is_absolute() and file != root and root in file.parents, "path_outside_owned_root")
    require(file == Path(os.path.abspath(file)), "path_not_canonical")
    for parent in reversed(file.parents):
        if parent == root or root in parent.parents:
            item = parent.lstat()
            require(stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode)
                    and item.st_uid == os.getuid(), "owned_ancestor_changed")
    item = file.lstat()
    require(stat.S_ISREG(item.st_mode) and not stat.S_ISLNK(item.st_mode)
            and item.st_uid == os.getuid() and stat.S_IMODE(item.st_mode) == 0o600,
            "owned_file_type_or_mode")
    return item


def root_guard(root, nonce, original, marker_original):
    item = root.lstat()
    require(stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode)
            and item.st_uid == os.getuid() and stat.S_IMODE(item.st_mode) == 0o700
            and item.st_dev == original.st_dev and item.st_ino == original.st_ino
            and root.resolve(strict=True) == root, "root_identity_changed")
    marker = root / ".owner"
    current = owned_path(root, marker)
    require(current.st_dev == marker_original.st_dev and current.st_ino == marker_original.st_ino
            and current.st_size == 32, "owner_marker_identity_changed")
    fd = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        require(identity(os.fstat(fd)) == identity(current)
                and os.read(fd, 33) == nonce.encode("ascii"), "owner_marker_changed")
    finally:
        os.close(fd)


class Cursor:
    def __init__(self, fd, size):
        self.fd, self.size, self.position = fd, size, 0
        self.entry_digest = None
        self.metadata_digest = None
        self.metadata_end = None
        self.descriptor_end = None
        self.values = 0

    def take(self, count):
        clock_check()
        require(0 <= count <= LIMITS["streamBytes"] and self.position + count <= self.size,
                "read_outside_input_or_chunk_bound")
        if self.metadata_end is not None:
            require(self.position + count <= self.metadata_end, "metadata_encoded_bound")
        if self.descriptor_end is not None:
            require(self.position + count <= self.descriptor_end, "descriptor_region_bound")
        output = os.pread(self.fd, count, self.position)
        require(len(output) == count, "input_truncated_during_parse")
        self.position += count
        if self.entry_digest is not None:
            self.entry_digest.update(output)
        if self.metadata_digest is not None:
            self.metadata_digest.update(output)
        return output

    def number(self, code):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def begin(self):
        self.entry_digest = hashlib.sha256()
        return self.position

    def finish(self, start):
        value = {"offset": start, "encodedBytes": self.position - start,
                 "encodedSha256": self.entry_digest.hexdigest()}
        self.entry_digest = None
        return value

    def string(self, maximum, retain=False):
        length = self.number("Q")
        require(length <= maximum, "gguf_string_bound")
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        remaining, chunks = length, []
        while remaining:
            chunk = self.take(min(remaining, LIMITS["streamBytes"]))
            decoder.decode(chunk, final=False)
            if retain:
                chunks.append(chunk)
            remaining -= len(chunk)
        decoder.decode(b"", final=True)
        return (b"".join(chunks).decode("utf-8") if retain else None), length

    def value(self, kind, retain=False):
        self.values += 1
        require(self.values <= LIMITS["metadataValues"], "metadata_value_count_bound")
        if kind in NUMERIC:
            value = self.number(NUMERIC[kind])
            require(kind != 7 or value in (0, 1), "invalid_gguf_boolean")
            return (bool(value) if kind == 7 else value), None, 1
        if kind == 8:
            # Strings are decoded only to validate UTF-8; tokenizer values never
            # enter the report. An allowlisted scalar can be retained below.
            value, _ = self.string(LIMITS["stringBytes"], retain=retain)
            return value, None, 1
        require(kind == 9, "unknown_metadata_value_kind")
        array_kind, count = self.number("I"), self.number("Q")
        require(count <= LIMITS["arrayItems"], "metadata_array_count_bound")
        require(array_kind != 9, "nested_array_outside_observer_profile")
        require(array_kind in NUMERIC or array_kind == 8, "unknown_metadata_array_kind")
        require(self.values + count <= LIMITS["metadataValues"], "metadata_value_count_bound")
        for _ in range(count):
            self.value(array_kind, retain=False)
        return None, array_kind, count


def raw_base64(data):
    return {"encoding": "base64", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "data": base64.b64encode(data).decode("ascii")}


def observe(fd, size, result):
    cursor = Cursor(fd, size)
    header = cursor.take(24)
    magic, version, tensor_count, metadata_count = struct.unpack("<4sIQQ", header)
    result["header"] = {"raw": raw_base64(header), "magicHex": magic.hex(), "littleEndianVersion": version,
                        "tensorCount": tensor_count, "metadataCount": metadata_count}
    require(magic == b"GGUF", "header_magic_refused")
    require(version == 3, "header_version_or_endianness_refused")
    require(1 <= tensor_count <= LIMITS["tensors"], "tensor_count_outside_observer_profile")
    require(metadata_count <= LIMITS["metadataEntries"], "metadata_count_outside_observer_profile")
    result["header"]["version3LittleEndianObserved"] = True
    cursor.metadata_end = 24 + LIMITS["metadataEncodedBytes"]
    cursor.metadata_digest = hashlib.sha256()
    keys, scalars = set(), {}
    for ordinal in range(metadata_count):
        start = cursor.begin()
        key, _ = cursor.string(LIMITS["keyBytes"], retain=True)
        require(key not in keys and re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", key, re.ASCII),
                "metadata_key_invalid_or_duplicate")
        keys.add(key)
        kind = cursor.number("I")
        retain = key in SCALAR_KEYS and kind != 9
        value, array_kind, count = cursor.value(kind, retain=retain)
        entry = {"ordinal": ordinal, "key": key, "kind": kind, "arrayKind": array_kind,
                 "count": count, **cursor.finish(start), "valueRetained": False}
        if retain and (not isinstance(value, str) or len(value.encode("utf-8")) <= LIMITS["retainedScalarStringBytes"]):
            require(not isinstance(value, float) or math.isfinite(value), "allowlisted_scalar_nonfinite")
            if isinstance(value, str):
                require(not any(ord(char) < 32 or ord(char) == 127 for char in value), "allowlisted_scalar_controls")
            entry["scalarValue"] = str(value) if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1 else value
            entry["scalarRepresentation"] = "decimal-string" if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1 else "json"
            entry["valueRetained"] = True
            scalars[key] = {"kind": kind, "value": entry["scalarValue"]}
        result["metadata"].append(entry)
    result["metadataRegion"] = {"offset": 24, "bytes": cursor.position - 24,
                                "sha256": cursor.metadata_digest.hexdigest(), "rawRetained": False,
                                "decodedValuesVisited": cursor.values}
    result["allowlistedScalars"] = scalars
    cursor.metadata_digest = cursor.metadata_end = None
    if "general.alignment" in keys:
        require("general.alignment" in scalars and scalars["general.alignment"]["kind"] == 4
                and type(scalars["general.alignment"]["value"]) is int,
                "alignment_metadata_type_or_retention")
        alignment = scalars["general.alignment"]
    else:
        alignment = {"kind": 4, "value": 32}
    require(alignment["kind"] == 4 and type(alignment["value"]) is int
            and 8 <= alignment["value"] <= LIMITS["alignment"] and alignment["value"] % 8 == 0,
            "alignment_outside_observer_profile")
    alignment_value = alignment["value"]
    descriptor_start = cursor.position
    cursor.descriptor_end = descriptor_start + LIMITS["descriptorBytes"]
    names = set()
    for ordinal in range(tensor_count):
        start = cursor.begin()
        name, _ = cursor.string(LIMITS["tensorNameBytes"], retain=True)
        require(name and name not in names and not any(ord(char) < 32 or ord(char) == 127 for char in name),
                "tensor_name_invalid_or_duplicate")
        names.add(name)
        rank = cursor.number("I")
        require(1 <= rank <= LIMITS["dimensions"], "tensor_rank_outside_observer_profile")
        dimensions = [cursor.number("Q") for _ in range(rank)]
        require(all(value > 0 for value in dimensions), "tensor_dimension_zero")
        ggml_type, offset = cursor.number("I"), cursor.number("Q")
        require(offset % alignment_value == 0, "tensor_offset_misaligned")
        result["tensors"].append({"ordinal": ordinal, "name": name, "dimensions": dimensions,
                                  "ggmlType": ggml_type, "relativeOffset": offset,
                                  "descriptor": cursor.finish(start)})
    descriptor_length = cursor.position - descriptor_start
    raw = os.pread(fd, descriptor_length, descriptor_start)
    require(len(raw) == descriptor_length, "descriptor_region_truncated")
    digest = hashlib.sha256(raw).hexdigest()
    zipped = gzip.compress(raw, mtime=0)
    require(len(zipped) <= LIMITS["descriptorBytes"] + 1024, "descriptor_gzip_bound")
    result["descriptorRegion"] = {"offset": descriptor_start, "bytes": descriptor_length, "sha256": digest,
                                  "raw": {"encoding": "gzip-base64", "bytes": descriptor_length, "sha256": digest,
                                          "compressedBytes": len(zipped), "data": base64.b64encode(zipped).decode("ascii")}}
    data_start = cursor.position + (-cursor.position % alignment_value)
    require(data_start <= size, "tensor_data_start_outside_input")
    for tensor in result["tensors"]:
        require(tensor["relativeOffset"] <= size - data_start, "tensor_start_outside_input")
        tensor["absoluteDataOffset"] = data_start + tensor["relativeOffset"]
    result["tensorData"] = {"offset": data_start, "alignment": alignment_value,
                            "alignmentSource": "metadata" if "general.alignment" in keys else "spec_default",
                            "payloadBytesReadForDecoding": 0, "payloadLengthsInferred": False,
                            "payloadSpanValidation": "not_performed_without_encoding_sizes"}


class Arguments(argparse.ArgumentParser):
    def error(self, message):
        raise Refusal("invalid_cli_arguments")


def main():
    source_commit = os.environ.get("SOURCE_COMMIT", "")
    result = {"schemaVersion": 1, "sourceCommit": source_commit if re.fullmatch(r"[a-f0-9]{40}", source_commit) else None,
              "status": "refused", "fixture": None, "observerSpec": SPEC, "limits": LIMITS,
              "applicationAdmission": "not_evaluated", "pairRestorationAdmission": "not_evaluated",
              "header": None, "metadata": [], "metadataRegion": None, "allowlistedScalars": {},
              "tensors": [], "descriptorRegion": None, "tensorData": None, "errors": [],
              "checks": {"rootIdentityVerified": False, "sourceUnchanged": False, "inputUnchanged": False,
                         "inputDescriptorClosed": False},
              "proofLimits": [
                  "Independent standard-library header/metadata/descriptors only; no application import, tokenizer/model execution, tensor-weight decoding or inference.",
                  "Full-file hashes read opaque input bytes, but no weight payload or non-allowlisted metadata value is retained. Tokenizer string values and metadata-region originals are excluded.",
                  "Numeric GGML encodings, including types unknown to Quantopsy, are observed without inferred byte lengths. Payload lengths, overlaps, decoding and model compatibility are not established.",
                  "This bounded little-endian version-3 observer refuses other versions/endianness, nested metadata arrays and declared limits; a refusal need not mean that the broader GGUF specification is invalid.",
                  "Before/after pathname, descriptor, root and full-file hash checks detect changes at their boundaries; they do not provide a filesystem snapshot. The supervisor owns process deadlines and artifact disposal.",
              ]}
    fd = None
    root = file = source = None
    original = marker_original = input_original = source_original = None
    source_fd = None

    def interrupted(number, frame):
        raise Refusal("inventory_deadline" if number == signal.SIGALRM else "inventory_interrupted")

    for number in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)
    signal.alarm(LIMITS["workSeconds"])
    try:
        require(os.name == "posix" and sys.platform == "linux" and sys.version_info >= (3, 11)
                and sys.flags.isolated and sys.dont_write_bytecode and os.environ.get("CI") == "true"
                and result["sourceCommit"] is not None, "approved_isolated_ci_required")
        parser = Arguments(add_help=False)
        parser.add_argument("--fixture", required=True, choices=sorted(FIXTURES))
        parser.add_argument("--input", required=True)
        args = parser.parse_args()
        result["fixture"] = args.fixture
        expected = FIXTURES[args.fixture]
        result["expectedArtifact"] = expected
        nonce = os.environ.get("QUANTOPSY_ARTIFACT_NONCE", "")
        raw_root = os.environ.get("QUANTOPSY_ARTIFACT_RUN_ROOT", "")
        require(re.fullmatch(r"[a-f0-9]{32}", nonce) and Path(raw_root).is_absolute(), "owned_root_arguments")
        root = Path(raw_root)
        require(root.resolve(strict=True) == root, "root_not_canonical")
        original, marker_original = root.lstat(), (root / ".owner").lstat()
        root_guard(root, nonce, original, marker_original)
        source = Path(__file__)
        source_original = owned_path(root, source)
        require(source_original.st_size <= LIMITS["sourceBytes"], "observer_source_bound")
        source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        require(identity(os.fstat(source_fd)) == identity(source_original), "observer_source_changed")
        source_hash = fingerprint(source_fd, source_original.st_size)
        require(re.fullmatch(r"[a-f0-9]{64}", os.environ.get("QUANTOPSY_ARTIFACT_INVENTORY_SHA256", ""))
                and source_hash == os.environ["QUANTOPSY_ARTIFACT_INVENTORY_SHA256"], "observer_source_pin")
        result["source"] = {"relativePath": str(source.relative_to(root)), "before": identity(source_original), "sha256": source_hash}
        file = Path(args.input)
        input_original = owned_path(root, file)
        require(input_original.st_size == expected["bytes"], "public_input_size_mismatch")
        fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW)
        require(identity(os.fstat(fd)) == identity(input_original), "input_changed_at_open")
        before_hash = fingerprint(fd, expected["bytes"])
        result["input"] = {"relativePath": str(file.relative_to(root)), "before": identity(input_original), "sha256Before": before_hash}
        require(before_hash == expected["sha256"], "public_input_hash_mismatch")
        observe(fd, expected["bytes"], result)
    except Refusal as error:
        result["errors"].append(str(error))
    except (OSError, UnicodeError, ValueError, struct.error, OverflowError):
        result["errors"].append("inventory_io_or_encoding_refusal")
    except Exception:
        result["errors"].append("inventory_internal_refusal")
    finally:
        try:
            if fd is not None:
                current = owned_path(root, file)
                after_hash = fingerprint(fd, input_original.st_size)
                result["input"]["after"] = identity(current)
                result["input"]["sha256After"] = after_hash
                require(identity(current) == identity(input_original) == identity(os.fstat(fd))
                        and after_hash == result["input"]["sha256Before"] == FIXTURES[result["fixture"]]["sha256"],
                        "input_changed_after_observation")
                result["checks"]["inputUnchanged"] = True
            if source_fd is not None:
                current_source = owned_path(root, source)
                after_source_hash = fingerprint(source_fd, source_original.st_size)
                result["source"]["after"] = identity(current_source)
                require(identity(current_source) == identity(source_original) == identity(os.fstat(source_fd))
                        and after_source_hash == result["source"]["sha256"], "observer_source_changed_after_observation")
                result["checks"]["sourceUnchanged"] = True
            if original is not None:
                root_guard(root, nonce, original, marker_original)
                result["checks"]["rootIdentityVerified"] = True
        except Refusal as error:
            result["errors"].append(str(error))
        except Exception:
            result["errors"].append("final_identity_check_refused")
        finally:
            for descriptor in (fd, source_fd):
                if descriptor is not None:
                    os.close(descriptor)
            result["checks"]["inputDescriptorClosed"] = fd is not None
            signal.alarm(0)
    if not result["errors"] and all(result["checks"].values()) and result["descriptorRegion"] is not None:
        result["status"] = "observed"
    result["elapsedMs"] = round((time.monotonic() - START) * 1000)
    output = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(output.encode("utf-8")) > LIMITS["reportBytes"]:
        result = {"schemaVersion": 1, "sourceCommit": result["sourceCommit"], "fixture": result["fixture"],
                  "status": "refused", "applicationAdmission": "not_evaluated", "pairRestorationAdmission": "not_evaluated",
                  "errors": ["inventory_report_bound"], "checks": result["checks"], "header": result["header"]}
        output = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    signal.signal(signal.SIGALRM, lambda number, frame: os._exit(2))
    signal.alarm(LIMITS["stdoutSeconds"])
    sys.stdout.write("QUANTOPSY_PUBLIC_GGUF_INVENTORY_RESULT=" + output + "\n")
    sys.stdout.flush()
    signal.alarm(0)
    return 0 if result["status"] == "observed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
