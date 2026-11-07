#!/usr/bin/env python3
"""Owned-CI opaque payload bindings; no application import or numerical decode."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
import types


PREFIX = "QUANTOPSY_PUBLIC_GGUF_V2_BINDINGS_RESULT="
INVENTORY_SHA256 = "e1a51e561fc9ba2ab0a2b4e3604575accd37d2b560eebd7cda62772bda32c1f5"
INVENTORY_BYTES = 22524
SELECTED = "blk.0.attn_q.weight"
Q8_NAME = "output.weight"
REVISION = "4cd12190a40543c61bad59a1b8d982dfc48de020"
BASE_URL = "https://huggingface.co/ggml-org/models-moved/resolve/" + REVISION + "/tinyllamas/"
INPUTS = (
    {"fixture": "stories15m-reference", "name": "stories15M.gguf", "bytes": 98357920,
     "sha256": "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a"},
    {"fixture": "stories15m-q4_0", "name": "stories15M-q4_0.gguf", "bytes": 19077344,
     "sha256": "6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04"},
)
QUANTIZATION_KEYS = {"general.file_type", "general.quantization_version",
                     "general.quantized_by", "general.quantized_at"}
PLAN_KEYS = {"kind", "schema_version", "reference", "quantized", "architecture", "provenance", "selected",
             "added_weight_bytes", "max_added_bytes", "max_elements", "lineage", "behavioral_validation"}
SELECTED_KEYS = {"name", "dimensions", "elements", "reference_kind", "quantized_kind", "reference_sha256",
                 "quantized_sha256", "reference_bytes", "quantized_bytes"}
PROVENANCE_KEYS = {"model_id", "revision", "license", "reference_source", "quantized_source"}
MANIFEST_KEYS = {"kind", "schema_version", "plan_sha256", "reference", "quantized", "output", "architecture",
                 "selected", "added_weight_bytes", "max_added_bytes", "max_elements", "reserve_bytes",
                 "metadata_omissions", "provenance", "lineage", "behavioral_validation", "verification"}
LIMITS = {"workSeconds": 90, "stdoutSeconds": 2, "reportBytes": 2 * 1024**2,
          "jsonBytes": 2 * 1024**2, "jsonDepth": 16, "jsonValues": 100000,
          "sourceBytes": 128 * 1024, "candidateBytes": 32 * 1024**2,
          "streamBytes": 1024**2, "selectedElements": 64000000, "outputBytes": 16 * 1024**3}
START = time.monotonic()


class Refusal(Exception):
    """A finite non-payload refusal code."""


def require(condition, code):
    if not condition:
        raise Refusal(code)


def clock_check():
    require(time.monotonic() - START < LIMITS["workSeconds"], "bindings_deadline")


def identity(value):
    return {"device": str(value.st_dev), "inode": str(value.st_ino), "uid": value.st_uid,
            "mode": format(stat.S_IMODE(value.st_mode), "04o"), "bytes": value.st_size,
            "mtimeNs": str(value.st_mtime_ns), "ctimeNs": str(value.st_ctime_ns)}


def owned_file(root, path):
    require(path.is_absolute() and root in path.parents
            and path == Path(os.path.abspath(path)), "owned_path_invalid")
    for parent in reversed(path.parents):
        if parent == root or root in parent.parents:
            value = parent.lstat()
            require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.getuid()
                    and stat.S_IMODE(value.st_mode) == 0o700, "owned_directory_changed")
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and value.st_uid == os.getuid()
            and stat.S_IMODE(value.st_mode) == 0o600, "owned_file_changed")
    return value


def root_guard(root, nonce, original, marker_original):
    value = root.lstat()
    require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.getuid()
            and stat.S_IMODE(value.st_mode) == 0o700
            and (value.st_dev, value.st_ino) == (original.st_dev, original.st_ino)
            and root.resolve(strict=True) == root, "root_identity_changed")
    marker = root / ".owner"
    current = owned_file(root, marker)
    require(identity(current) == identity(marker_original) and current.st_size == 32,
            "owner_identity_changed")
    descriptor = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        require(identity(os.fstat(descriptor)) == identity(current)
                and os.read(descriptor, 33) == nonce.encode("ascii"), "owner_nonce_changed")
    finally:
        os.close(descriptor)


def hash_span(descriptor, offset, length):
    digest = hashlib.sha256()
    while length:
        clock_check()
        chunk = os.pread(descriptor, min(length, LIMITS["streamBytes"]), offset)
        require(bool(chunk), "opaque_span_truncated")
        digest.update(chunk)
        offset += len(chunk)
        length -= len(chunk)
    return digest.hexdigest()


class HeldFile:
    def __init__(self, root, relative, maximum, records, opened, *, size=None, sha256=None):
        self.fd = None
        self.root, self.path = root, root / relative
        self.before = owned_file(root, self.path)
        require(0 < self.before.st_size <= maximum
                and (size is None or self.before.st_size == size), "owned_file_size_refused")
        self.record = {"relativePath": relative, "before": identity(self.before),
                       "sha256Before": None, "sha256After": None, "unchanged": False, "closed": False}
        records.append(self.record)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        opened.append(self)
        require(identity(os.fstat(self.fd)) == identity(self.before), "file_changed_at_open")
        self.record["sha256Before"] = hash_span(self.fd, 0, self.before.st_size)
        require(os.pread(self.fd, 1, self.before.st_size) == b"", "file_grew_at_open")
        require(sha256 is None or self.record["sha256Before"] == sha256, "owned_file_hash_refused")

    def read(self, maximum):
        require(self.before.st_size <= maximum, "retained_file_bound")
        value = os.pread(self.fd, self.before.st_size, 0)
        require(len(value) == self.before.st_size, "retained_file_truncated")
        return value

    def verify(self):
        current = owned_file(self.root, self.path)
        self.record["after"] = identity(current)
        self.record["sha256After"] = hash_span(self.fd, 0, self.before.st_size)
        require(identity(current) == identity(self.before) == identity(os.fstat(self.fd))
                and self.record["sha256After"] == self.record["sha256Before"]
                and os.pread(self.fd, 1, self.before.st_size) == b"", "opened_file_changed")
        self.record["unchanged"] = True

    def close(self):
        if self.fd is not None:
            descriptor, self.fd = self.fd, None
            os.close(descriptor)
            self.record["closed"] = True


def load_observer(held):
    require(held.before.st_size == INVENTORY_BYTES
            and held.record["sha256Before"] == INVENTORY_SHA256, "observer_source_pin")
    source = held.read(LIMITS["sourceBytes"])
    require(hashlib.sha256(source).hexdigest() == INVENTORY_SHA256, "observer_compiled_bytes_pin")
    module = types.ModuleType("_pinned_public_gguf_inventory")
    module.__file__ = str(held.path)
    exec(compile(source, str(held.path), "exec"), module.__dict__)
    for expected in INPUTS:
        actual = module.FIXTURES[expected["fixture"]]
        require(actual == {"bytes": expected["bytes"], "sha256": expected["sha256"],
                           "revision": REVISION, "url": BASE_URL + expected["name"]}, "observer_fixture_pin")
    return module


def observe_payloads(observer, held):
    inventory = {"metadata": [], "tensors": []}
    observer.observe(held.fd, held.before.st_size, inventory)
    data_offset = inventory["tensorData"]["offset"]
    alignment = inventory["tensorData"]["alignment"]
    spans = []
    for tensor in inventory["tensors"]:
        elements = 1
        for value in tensor["dimensions"]:
            require(type(value) is int and value > 0
                    and elements <= held.before.st_size * 32 // 18 // value, "tensor_elements_bound")
            elements *= value
        kind = tensor["ggmlType"]
        require(kind in (0, 1, 2, 8), "tensor_encoding_not_known")
        if kind in (0, 1):
            length = elements * (4 if kind == 0 else 2)
        else:
            require(tensor["dimensions"][0] % 32 == 0, "quantized_row_incomplete")
            length = elements // 32 * (18 if kind == 2 else 34)
        start = tensor["absoluteDataOffset"]
        require(start == data_offset + tensor["relativeOffset"]
                and tensor["relativeOffset"] % alignment == 0
                and 0 <= start <= held.before.st_size and length <= held.before.st_size - start,
                "tensor_span_outside_file")
        tensor.update({"elements": elements, "payloadBytes": length})
        spans.append((start, start + length))
    previous = data_offset
    for start, end in sorted(spans):
        require(start >= previous, "tensor_payload_overlap")
        previous = end
    for tensor in inventory["tensors"]:
        tensor["payloadSha256"] = hash_span(held.fd, tensor["absoluteDataOffset"], tensor["payloadBytes"])
    if any(tensor["ggmlType"] in (2, 8) for tensor in inventory["tensors"]):
        require(inventory["allowlistedScalars"].get("general.quantization_version") == {"kind": 4, "value": 2},
                "quantization_version_policy")
    inventory["tensorData"]["payloadSpanValidation"] = "validated_explicit_supported_block_sizes"
    inventory["payloadEvidence"] = {"allSpansValidated": True, "allPayloadsHashed": True,
                                    "payloadsDecoded": False, "sizeSource": "explicit_ggml_0_1_2_8_blocks"}
    return inventory


def metadata_signature(row):
    return row["kind"], row["arrayKind"], row["count"], row["encodedBytes"], row["encodedSha256"]


def by_name(inventory):
    return {row["name"]: row for row in inventory["tensors"]}


def selected_binding(reference, quantized):
    a, b = reference[SELECTED], quantized[SELECTED]
    return {"name": SELECTED, "dimensions": a["dimensions"], "elements": a["elements"],
            "reference_kind": a["ggmlType"], "quantized_kind": b["ggmlType"],
            "reference_sha256": a["payloadSha256"], "quantized_sha256": b["payloadSha256"],
            "reference_bytes": a["payloadBytes"], "quantized_bytes": b["payloadBytes"]}


def verify_pair(reference, quantized):
    a, b = by_name(reference), by_name(quantized)
    require(len(a) == len(b) == 57 and set(a) == set(b), "pair_tensor_inventory_mismatch")
    require(all(row["ggmlType"] in (0, 1) for row in a.values()), "reference_encoding_mismatch")
    require(all(a[name]["dimensions"] == b[name]["dimensions"] for name in a), "pair_dimensions_mismatch")
    require(reference["allowlistedScalars"].get("general.architecture") == {"kind": 8, "value": "llama"}
            and quantized["allowlistedScalars"].get("general.architecture") == {"kind": 8, "value": "llama"},
            "pair_architecture_mismatch")
    am = {row["key"]: metadata_signature(row) for row in reference["metadata"] if row["key"] not in QUANTIZATION_KEYS}
    bm = {row["key"]: metadata_signature(row) for row in quantized["metadata"] if row["key"] not in QUANTIZATION_KEYS}
    require(am == bm, "pair_encoded_metadata_mismatch")
    unquantized = []
    for name, row in b.items():
        if row["ggmlType"] in (0, 1):
            original = a[name]
            require((row["ggmlType"], row["payloadBytes"], row["payloadSha256"])
                    == (original["ggmlType"], original["payloadBytes"], original["payloadSha256"]),
                    "unquantized_payload_mismatch")
            unquantized.append({"name": name, "ggmlType": row["ggmlType"], "bytes": row["payloadBytes"],
                                "referenceSha256": original["payloadSha256"], "quantizedSha256": row["payloadSha256"]})
    require(len(unquantized) == 13, "public_fixture_unquantized_count_mismatch")
    require(a[SELECTED]["dimensions"] == b[SELECTED]["dimensions"] == [288, 288]
            and a[SELECTED]["ggmlType"] == 0 and b[SELECTED]["ggmlType"] == 2, "selected_tensor_mismatch")
    q8 = b.get(Q8_NAME)
    require(q8 is not None and q8["ggmlType"] == 8 and q8["dimensions"] == [288, 32000]
            and q8["payloadBytes"] == 9792000
            and sum(row["ggmlType"] == 8 for row in b.values()) == 1, "public_fixture_q8_mismatch")
    return {"matched": True, "tensorCount": 57, "metadataOutsideFourQuantizationKeysMatched": True,
            "unquantizedCount": len(unquantized), "unquantizedBindings": unquantized,
            "selected": selected_binding(a, b),
            "opaqueQ8": {"name": Q8_NAME, "dimensions": q8["dimensions"], "ggmlType": 8,
                         "bytes": q8["payloadBytes"], "sha256": q8["payloadSha256"]},
            "selectedFiniteValues": "not_evaluated", "lineage": "unverified", "behavioralValidation": "not_run"}


def reject_constant(value):
    raise Refusal("json_nonfinite")


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "json_duplicate_key")
        value[key] = item
    return value


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(held):
    raw = held.read(LIMITS["jsonBytes"])
    value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=unique_object,
                       parse_constant=reject_constant)
    pending, visited = [(value, 0)], 0
    while pending:
        current, depth = pending.pop()
        visited += 1
        require(depth <= LIMITS["jsonDepth"] and visited <= LIMITS["jsonValues"], "json_structure_bound")
        if type(current) is dict:
            for key, item in current.items():
                key.encode("utf-8", errors="strict")
                pending.append((item, depth + 1))
        elif type(current) is list:
            pending.extend((item, depth + 1) for item in current)
        elif type(current) is str:
            current.encode("utf-8", errors="strict")
        elif type(current) is float:
            require(math.isfinite(current), "json_nonfinite")
        else:
            require(current is None or type(current) in (bool, int), "json_type_refused")
    require(canonical(value) == raw, "json_not_canonical")
    return value, raw


def whole(value, low, high):
    return type(value) is int and low <= value <= high


def verify_candidate(reference, quantized, candidate, pair, external, bundled, manifest, candidate_file):
    plan, plan_raw = read_json(external)
    bundled_plan, bundled_raw = read_json(bundled)
    receipt, receipt_raw = read_json(manifest)
    require(plan_raw == bundled_raw and plan == bundled_plan, "plan_files_differ")
    require(type(plan) is dict and set(plan) == PLAN_KEYS and type(plan["schema_version"]) is int
            and plan["schema_version"] == 2 and plan["kind"] == "quantopsy.repair-plan"
            and plan["architecture"] == "llama", "plan_v2_contract_mismatch")
    require(type(receipt) is dict and set(receipt) == MANIFEST_KEYS and type(receipt["schema_version"]) is int
            and receipt["schema_version"] == 2 and receipt["kind"] == "quantopsy.repair-result", "manifest_v2_contract_mismatch")
    for key, expected in zip(("reference", "quantized"), INPUTS):
        require(plan[key] == {"name": expected["name"], "size": expected["bytes"], "sha256": expected["sha256"]}
                and receipt[key] == plan[key], "record_input_binding_mismatch")
    require(type(plan["provenance"]) is dict and set(plan["provenance"]) == PROVENANCE_KEYS
            and all(type(value) is str and value.strip() and "\0" not in value and len(value.encode("utf-8")) <= 2048
                    for value in plan["provenance"].values()), "declared_provenance_mismatch")
    require(type(plan["selected"]) is list and len(plan["selected"]) == 1
            and type(plan["selected"][0]) is dict and set(plan["selected"][0]) == SELECTED_KEYS
            and plan["selected"][0] == pair["selected"], "plan_selected_binding_mismatch")
    require(all(type(plan["selected"][0][key]) is int for key in ("elements", "reference_kind", "quantized_kind", "reference_bytes", "quantized_bytes"))
            and all(type(value) is int for value in plan["selected"][0]["dimensions"]), "plan_selected_integer_types")
    growth = pair["selected"]["reference_bytes"] - pair["selected"]["quantized_bytes"]
    require(whole(plan["added_weight_bytes"], 0, LIMITS["outputBytes"]) and plan["added_weight_bytes"] == growth
            and whole(plan["max_added_bytes"], growth, LIMITS["outputBytes"])
            and whole(plan["max_elements"], pair["selected"]["elements"], LIMITS["selectedElements"]), "plan_budgets_mismatch")
    require(plan["lineage"] == "operator_declared_compatibility_checked" and plan["behavioral_validation"] == "not_run",
            "plan_evidence_claim_mismatch")
    for key in (PLAN_KEYS - {"kind", "schema_version"}):
        require(canonical(receipt[key]) == canonical(plan[key]), "manifest_plan_binding_mismatch")
    require(receipt["plan_sha256"] == hashlib.sha256(plan_raw).hexdigest()
            and whole(receipt["reserve_bytes"], 0, 2**53 - 1), "manifest_plan_hash_or_reserve_mismatch")
    require(receipt["output"] == {"name": "candidate.gguf", "size": candidate_file.before.st_size,
                                   "sha256": candidate_file.record["sha256Before"]}, "manifest_output_binding_mismatch")
    require(receipt["verification"] == {"container_reparsed": True, "metadata_preserved": True,
                                         "tensor_payloads_verified": True, "source_hashes_verified": True}
            and all(type(value) is bool for value in receipt["verification"].values()), "manifest_verification_mismatch")
    qm = [row for row in quantized["metadata"] if row["key"] != "general.file_type"]
    cm = candidate["metadata"]
    require([row["key"] for row in cm] == [row["key"] for row in qm]
            and all(metadata_signature(a) == metadata_signature(b) for a, b in zip(cm, qm)), "candidate_metadata_mismatch")
    omissions = ["general.file_type"] if any(row["key"] == "general.file_type" for row in quantized["metadata"]) else []
    require(receipt["metadata_omissions"] == omissions, "manifest_omissions_mismatch")
    require(candidate["tensorData"]["alignment"] == quantized["tensorData"]["alignment"]
            and len(candidate["tensors"]) == 57
            and [row["name"] for row in candidate["tensors"]] == [row["name"] for row in quantized["tensors"]],
            "candidate_inventory_order_mismatch")
    a, b = by_name(reference), by_name(quantized)
    bindings = []
    for actual in candidate["tensors"]:
        name = actual["name"]
        prescribed = a[name] if name == SELECTED else b[name]
        require((actual["dimensions"], actual["ggmlType"], actual["payloadBytes"], actual["payloadSha256"])
                == (prescribed["dimensions"], prescribed["ggmlType"], prescribed["payloadBytes"], prescribed["payloadSha256"]),
                "candidate_payload_binding_mismatch")
        bindings.append({"name": name, "source": "reference" if name == SELECTED else "quantized",
                         "ggmlType": actual["ggmlType"], "bytes": actual["payloadBytes"],
                         "expectedSha256": prescribed["payloadSha256"], "candidateSha256": actual["payloadSha256"]})
    return {"matched": True, "tensorCount": 57, "selectedCount": 1, "unselectedCount": 56,
            "payloadBindings": bindings, "planSha256": hashlib.sha256(plan_raw).hexdigest(),
            "manifestSha256": hashlib.sha256(receipt_raw).hexdigest(),
            "candidateSha256": candidate_file.record["sha256Before"], "candidateBytes": candidate_file.before.st_size,
            "declaredProvenanceSha256": hashlib.sha256(canonical(plan["provenance"])).hexdigest(),
            "metadataOrderAndBytesPreservedMinusFileType": True, "behavioralValidation": "not_run"}


class Arguments(argparse.ArgumentParser):
    def error(self, message):
        raise Refusal("invalid_cli_arguments")


def main():
    commit = os.environ.get("SOURCE_COMMIT", "")
    result = {"schemaVersion": 1, "sourceCommit": commit if re.fullmatch(r"[a-f0-9]{40}", commit) else None,
              "phase": None, "status": "refused", "productAccepted": False, "files": [], "inputs": [],
              "pair": None, "candidate": None, "errors": [], "limits": LIMITS,
              "provenance": {"observerSha256": INVENTORY_SHA256, "observerBytes": INVENTORY_BYTES,
                             "publicRevision": REVISION, "publicArtifacts": [dict(row, url=BASE_URL + row["name"]) for row in INPUTS],
                             "ggufSpec": "https://github.com/ggml-org/ggml/blob/facdc44a67da17df4433adcc468b424f48e9ded2/docs/gguf.md",
                             "q8BlockSpec": "https://github.com/ggml-org/llama.cpp/blob/eb4e244be77a906a183cd4a00fd51d9ad3200955/ggml/src/ggml-common.h#L251-L257"},
              "checks": {"rootIdentityVerified": False, "allOpenedFilesUnchanged": False,
                         "allDescriptorsClosed": False, "pairBindingsVerified": False, "candidateBindingsVerified": None},
              "proofLimits": [
                  "Opaque byte hashes and descriptor/metadata compatibility only; no numerical decode, application import, finite-value check, model inference or recovered-behavior proof.",
                  "The counts 57 tensors, 13 unquantized bindings and one selected matrix are specific to these two immutable public fixtures.",
                  "No weight payload, tokenizer value or metadata-region original is retained. The exact pinned independent observer retains only bounded headers, descriptor originals and encoded-metadata hashes.",
                  "Before/after descriptor/path/full-hash checks detect changes at their boundaries; they do not provide a filesystem snapshot or cryptographic model lineage.",
              ]}
    opened = []
    root = original = marker_original = None
    nonce = ""

    def interrupted(number, frame):
        raise Refusal("bindings_deadline" if number == signal.SIGALRM else "bindings_interrupted")

    for number in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)
    signal.alarm(LIMITS["workSeconds"])
    try:
        require(os.name == "posix" and sys.platform == "linux" and sys.version_info >= (3, 11)
                and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
                and os.environ.get("CI") == "true" and result["sourceCommit"] is not None, "approved_isolated_ci_required")
        parser = Arguments(add_help=False)
        parser.add_argument("--phase", required=True, choices=("pair", "candidate"))
        args = parser.parse_args()
        result["phase"] = args.phase
        result["checks"]["candidateBindingsVerified"] = False if args.phase == "candidate" else None
        nonce = os.environ.get("QUANTOPSY_ARTIFACT_NONCE", "")
        raw_root = os.environ.get("QUANTOPSY_ARTIFACT_RUN_ROOT", "")
        require(re.fullmatch(r"[a-f0-9]{32}", nonce) and Path(raw_root).is_absolute(), "owned_root_arguments")
        root = Path(raw_root)
        require(root.resolve(strict=True) == root, "root_not_canonical")
        original, marker_original = root.lstat(), (root / ".owner").lstat()
        root_guard(root, nonce, original, marker_original)
        own_hash = os.environ.get("QUANTOPSY_ARTIFACT_BINDINGS_SHA256", "")
        require(re.fullmatch(r"[a-f0-9]{64}", own_hash), "bindings_source_pin_missing")
        own_path = root / "helpers/public_gguf_v2_bindings.py"
        require(Path(__file__) == own_path, "bindings_source_path_mismatch")
        HeldFile(root, "helpers/public_gguf_v2_bindings.py", LIMITS["sourceBytes"], result["files"], opened, sha256=own_hash)
        inventory_source = HeldFile(root, "helpers/public_gguf_inventory.py", LIMITS["sourceBytes"], result["files"], opened,
                                    size=INVENTORY_BYTES, sha256=INVENTORY_SHA256)
        observer = load_observer(inventory_source)
        inventories = []
        for expected in INPUTS:
            held = HeldFile(root, "models/" + expected["name"], expected["bytes"], result["files"], opened,
                            size=expected["bytes"], sha256=expected["sha256"])
            row = {"fixture": expected["fixture"], "file": held.record, "inventory": {"metadata": [], "tensors": []}}
            result["inputs"].append(row)
            row["inventory"] = observe_payloads(observer, held)
            inventories.append(row["inventory"])
        result["pair"] = verify_pair(*inventories)
        result["checks"]["pairBindingsVerified"] = True
        if args.phase == "candidate":
            candidate_file = HeldFile(root, "output/candidate/candidate.gguf", LIMITS["candidateBytes"], result["files"], opened)
            candidate = {"file": candidate_file.record, "inventory": observe_payloads(observer, candidate_file)}
            result["candidate"] = candidate
            json_files = [HeldFile(root, relative, LIMITS["jsonBytes"], result["files"], opened) for relative in
                          ("output/repair-plan-v2.json", "output/candidate/plan.json", "output/candidate/repair.json")]
            candidate["bindings"] = verify_candidate(*inventories, candidate["inventory"], result["pair"], *json_files, candidate_file)
            result["checks"]["candidateBindingsVerified"] = True
    except Refusal as error:
        result["errors"].append(str(error))
    except Exception as error:
        if type(error).__name__ == "Refusal" and re.fullmatch(r"[a-z0-9_]{1,96}", str(error)):
            result["errors"].append("observer_" + str(error))
        else:
            result["errors"].append("bindings_io_encoding_or_contract_refusal")
    finally:
        for held in opened:
            try:
                held.verify()
            except Refusal as error:
                result["errors"].append(str(error))
            except Exception:
                result["errors"].append("final_file_check_refused")
        result["checks"]["allOpenedFilesUnchanged"] = bool(opened) and all(held.record["unchanged"] for held in opened)
        try:
            if original is not None and marker_original is not None:
                root_guard(root, nonce, original, marker_original)
                result["checks"]["rootIdentityVerified"] = True
        except Exception:
            result["errors"].append("final_root_check_refused")
        for held in reversed(opened):
            try:
                held.close()
            except Exception:
                result["errors"].append("descriptor_close_refused")
        result["checks"]["allDescriptorsClosed"] = bool(opened) and all(held.record["closed"] for held in opened)
        signal.alarm(0)
    checks = result["checks"]
    if (not result["errors"] and all(checks[key] is True for key in
                                    ("rootIdentityVerified", "allOpenedFilesUnchanged", "allDescriptorsClosed", "pairBindingsVerified"))
            and (result["phase"] == "pair" or checks["candidateBindingsVerified"] is True)):
        result["status"] = "verified"
    result["elapsedMs"] = round((time.monotonic() - START) * 1000)
    output = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(output.encode("utf-8")) > LIMITS["reportBytes"]:
        result = {"schemaVersion": 1, "sourceCommit": result["sourceCommit"], "phase": result["phase"],
                  "status": "refused", "productAccepted": False, "errors": ["bindings_report_bound"], "checks": checks}
        output = json.dumps(result, separators=(",", ":"))
    signal.signal(signal.SIGALRM, lambda number, frame: os._exit(2))
    signal.alarm(LIMITS["stdoutSeconds"])
    sys.stdout.write(PREFIX + output + "\n")
    sys.stdout.flush()
    signal.alarm(0)
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
