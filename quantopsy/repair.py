"""Compatibility-bound, create-only restoration of selected dense Llama weights."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import BinaryIO

from .contracts import DEFAULT_LIMITS, TYPE_F16, TYPE_F32, TYPE_Q4_0, TensorInfo
from .errors import QuantopsyError, fail
from .gguf import GGUFFile
from .io import canonical_json_bytes


PLAN_KIND = "quantopsy.repair-plan"
MANIFEST_KIND = "quantopsy.repair-result"
_QUANTIZATION_KEYS = frozenset({
    "general.file_type", "general.quantization_version",
    "general.quantized_by", "general.quantized_at",
})
_PROVENANCE_KEYS = frozenset({
    "model_id", "revision", "license", "reference_source", "quantized_source",
})
_TARGET = re.compile(r"blk\.[0-9]+\.(?:attn_(?:q|k|v|output)|ffn_(?:gate|up|down))\.weight", re.ASCII)
_UNSUPPORTED_LAYOUT = re.compile(r"expert|(?:^|[._])exps?(?:[._]|$)|moe|lora|adapter|mmproj|vision|audio|attn_qkv|ffn_gate_up", re.ASCII | re.IGNORECASE)
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_MAX_INTEGER = 2**53 - 1
_CHUNK = 1024**2


def _record(value: object, fields: set[str] | frozenset[str]) -> dict:
    if type(value) is not dict or any(type(key) is not str for key in value) or set(value) != fields:
        fail("repair_invalid_plan", "Repair records must contain exactly their supported fields.")
    return value


def _integer(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        fail("repair_invalid_budget", "Repair counts and budgets must be bounded whole numbers.")
    return value


def _text(value: object, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum or "\0" in value:
        fail("repair_invalid_plan", "Repair identity and provenance must be bounded nonblank text.")
    try:
        if len(value.encode("utf-8")) > maximum:
            fail("repair_invalid_plan", "Repair identity and provenance exceed their UTF-8 limit.")
    except UnicodeError:
        fail("repair_invalid_plan", "Repair text must be valid UTF-8.")
    return value


def _hash(value: object) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        fail("repair_invalid_plan", "Repair hashes must be complete lowercase SHA-256 values.")
    return value


def _provenance(value: object) -> dict[str, str]:
    raw = _record(value, _PROVENANCE_KEYS)
    return {key: _text(raw[key], 2048) for key in sorted(_PROVENANCE_KEYS)}


def _input_binding(value: object) -> dict:
    raw = _record(value, {"name", "sha256", "size"})
    name = _text(raw["name"], 255)
    if name in (".", "..") or any(character in name for character in ("/", "\\", "\n", "\r")):
        fail("repair_invalid_plan", "Input labels must be filenames without directory paths.")
    return {"name": name, "sha256": _hash(raw["sha256"]),
            "size": _integer(raw["size"], 1, DEFAULT_LIMITS.max_file_bytes)}


def _binding(source: GGUFFile) -> dict:
    return _input_binding({"name": source.path.name, "sha256": source.sha256(), "size": source.file_size})


def _dense_llama(source: GGUFFile, architecture: str) -> None:
    if architecture != "llama":
        fail("repair_architecture_unsupported", "Weight restoration currently supports dense Llama GGUF inputs only.")
    for key, entry in source.metadata.items():
        if "expert" in key or "moe" in key:
            if key.endswith((".expert_count", ".expert_used_count")) and type(entry.value) is int and entry.value == 0:
                continue
            fail("repair_layout_unsupported", "Expert metadata is outside the dense Llama repair profile.")
    for tensor in source.tensors:
        if len(tensor.dimensions) > 2 or _UNSUPPORTED_LAYOUT.search(tensor.name):
            fail("repair_layout_unsupported", "Expert, fused, adapter and multimodal tensor layouts are unsupported.")


def pair_inputs(reference: GGUFFile, quantized: GGUFFile, *, require_llama: bool = False) -> dict:
    """Check compatibility, without claiming lineage or taking descriptor ownership."""
    if type(require_llama) is not bool:
        fail("repair_invalid_plan", "The dense-Llama requirement must be a boolean.")
    reference.assert_unchanged()
    quantized.assert_unchanged()
    if any(tensor.kind not in (TYPE_F32, TYPE_F16) for tensor in reference.tensors):
        fail("repair_reference_unsupported", "The reference must contain only F32 and F16 tensors.")
    reference_keys = set(reference.metadata) - _QUANTIZATION_KEYS
    quantized_keys = set(quantized.metadata) - _QUANTIZATION_KEYS
    if reference_keys != quantized_keys or any(reference.metadata[key].encoded != quantized.metadata[key].encoded for key in reference_keys):
        fail("repair_pair_mismatch", "The paired model metadata does not match outside quantization-only fields.")
    architecture_entry = reference.metadata.get("general.architecture")
    if architecture_entry is None or architecture_entry.kind != 8 or type(architecture_entry.value) is not str or not architecture_entry.value:
        fail("repair_pair_mismatch", "The paired model architecture must be declared as text.")
    architecture = architecture_entry.value
    reference_tensors = {tensor.name: tensor for tensor in reference.tensors}
    quantized_tensors = {tensor.name: tensor for tensor in quantized.tensors}
    if not reference_tensors or set(reference_tensors) != set(quantized_tensors):
        fail("repair_pair_mismatch", "The paired inputs must contain the same complete tensor inventory.")
    for name, tensor in quantized_tensors.items():
        original = reference_tensors[name]
        if original.dimensions != tensor.dimensions:
            fail("repair_pair_mismatch", "Paired tensor dimensions must match exactly.")
        if tensor.kind in (TYPE_F32, TYPE_F16) and (
            tensor.kind != original.kind or quantized.hash_tensor(tensor) != reference.hash_tensor(original)
        ):
            fail("repair_pair_mismatch", "Unquantized tensors must keep their reference encoding and payload bytes.")
    if require_llama:
        _dense_llama(reference, architecture)
        _dense_llama(quantized, architecture)
    result = {"architecture": architecture, "tensor_count": len(reference_tensors),
              "reference": _binding(reference), "quantized": _binding(quantized),
              "lineage": "compatibility_checked_lineage_unverified"}
    reference.assert_unchanged()
    quantized.assert_unchanged()
    return result


def _selection(names: object) -> list[str]:
    if type(names) is not list or not 1 <= len(names) <= DEFAULT_LIMITS.max_tensors:
        fail("repair_selection_invalid", "Choose an explicit nonempty bounded list of repair tensors.")
    result = []
    seen = set()
    for name in names:
        if type(name) is not str or len(name) > DEFAULT_LIMITS.max_tensor_name_bytes or not _TARGET.fullmatch(name) or name in seen:
            fail("repair_selection_invalid", "Select each supported dense Llama matrix exactly once.")
        _text(name, DEFAULT_LIMITS.max_tensor_name_bytes)
        seen.add(name)
        result.append(name)
    return result


def validate_plan(value: object) -> dict:
    raw = _record(value, {"kind", "schema_version", "reference", "quantized", "architecture", "provenance", "selected",
                          "added_weight_bytes", "max_added_bytes", "max_elements", "lineage", "behavioral_validation"})
    if type(raw["kind"]) is not str or raw["kind"] != PLAN_KIND or type(raw["schema_version"]) is not int or raw["schema_version"] != 1 or type(raw["architecture"]) is not str or raw["architecture"] != "llama":
        fail("repair_invalid_plan", "Use the supported version-one dense Llama repair plan.")
    if type(raw["lineage"]) is not str or raw["lineage"] != "operator_declared_compatibility_checked" or type(raw["behavioral_validation"]) is not str or raw["behavioral_validation"] != "not_run":
        fail("repair_invalid_plan", "Repair plans cannot claim verified lineage or recovered behavior.")
    budget = _integer(raw["max_added_bytes"], 0, DEFAULT_LIMITS.max_output_bytes)
    work = _integer(raw["max_elements"], 1, DEFAULT_LIMITS.max_selected_elements)
    selected = raw["selected"]
    if type(selected) is not list or not 1 <= len(selected) <= DEFAULT_LIMITS.max_tensors:
        fail("repair_selection_invalid", "A plan must retain its complete explicit tensor selection.")
    rows = []
    total_elements = 0
    added_bytes = 0
    names = []
    for value in selected:
        row = _record(value, {"name", "dimensions", "elements", "reference_kind", "quantized_kind", "reference_sha256",
                              "quantized_sha256", "reference_bytes", "quantized_bytes"})
        names.append(row["name"])
        dimensions = row["dimensions"]
        if type(dimensions) is not list or len(dimensions) != 2:
            fail("repair_selection_invalid", "Repair tensors must be rank-two matrices.")
        shape = [_integer(item, 1, work) for item in dimensions]
        elements = _integer(row["elements"], 1, work)
        if shape[0] * shape[1] != elements or shape[0] % 32:
            fail("repair_selection_invalid", "Q4_0 repair matrices must contain whole 32-value rows.")
        if type(row["reference_kind"]) is not int or row["reference_kind"] != TYPE_F16 or type(row["quantized_kind"]) is not int or row["quantized_kind"] != TYPE_Q4_0:
            fail("repair_selection_invalid", "Restore Q4_0 tensors using matching F16 reference payloads.")
        original_bytes = _integer(row["reference_bytes"], 1, DEFAULT_LIMITS.max_file_bytes)
        encoded_bytes = _integer(row["quantized_bytes"], 1, DEFAULT_LIMITS.max_file_bytes)
        if original_bytes != elements * 2 or encoded_bytes != elements // 32 * 18:
            fail("repair_selection_invalid", "Planned byte counts must match the supported tensor encodings.")
        total_elements += elements
        added_bytes += original_bytes - encoded_bytes
        rows.append({"name": row["name"], "dimensions": shape, "elements": elements,
                     "reference_kind": TYPE_F16, "quantized_kind": TYPE_Q4_0,
                     "reference_sha256": _hash(row["reference_sha256"]), "quantized_sha256": _hash(row["quantized_sha256"]),
                     "reference_bytes": original_bytes, "quantized_bytes": encoded_bytes})
    _selection(names)
    if total_elements > work or added_bytes > budget or _integer(raw["added_weight_bytes"], 0, DEFAULT_LIMITS.max_output_bytes) != added_bytes:
        fail("repair_budget_exceeded", "The explicit selected work and added-weight budgets must cover this plan.")
    result = {"kind": PLAN_KIND, "schema_version": 1, "reference": _input_binding(raw["reference"]),
              "quantized": _input_binding(raw["quantized"]), "architecture": "llama", "provenance": _provenance(raw["provenance"]),
              "selected": rows, "added_weight_bytes": added_bytes, "max_added_bytes": budget, "max_elements": work,
              "lineage": "operator_declared_compatibility_checked", "behavioral_validation": "not_run"}
    canonical_json_bytes(result)
    return result


def _finite_values(source: GGUFFile, tensor: TensorInfo) -> None:
    count = 0
    for value in source.iter_values(tensor):
        count += 1
        if count > tensor.elements or not math.isfinite(value):
            fail("repair_invalid_values", "Selected tensor values must be finite and match their declared count.")
    if count != tensor.elements:
        fail("repair_invalid_values", "Selected tensor values must match their declared count.")


def create_plan(reference: GGUFFile, quantized: GGUFFile, names: list[str], provenance: dict,
                max_added_bytes: int, *, max_elements: int = DEFAULT_LIMITS.max_selected_elements) -> dict:
    selected_names = _selection(names)
    declared = _provenance(provenance)
    budget = _integer(max_added_bytes, 0, DEFAULT_LIMITS.max_output_bytes)
    work = _integer(max_elements, 1, min(DEFAULT_LIMITS.max_selected_elements, reference.limits.max_selected_elements, quantized.limits.max_selected_elements))
    pair = pair_inputs(reference, quantized, require_llama=True)
    tensors = []
    elements = 0
    added = 0
    # Validate the complete selection and work bound before decoding any values.
    for name in selected_names:
        original, tensor = reference.tensor(name), quantized.tensor(name)
        if len(tensor.dimensions) != 2 or original.kind != TYPE_F16 or tensor.kind != TYPE_Q4_0:
            fail("repair_selection_invalid", "Choose rank-two Q4_0 matrices with matching F16 references.")
        elements += tensor.elements
        added += original.nbytes - tensor.nbytes
        tensors.append((original, tensor))
    if elements > work or added > budget:
        fail("repair_budget_exceeded", "The explicit selected work or added-weight budget is too small.")
    rows = []
    for original, tensor in tensors:
        _finite_values(reference, original)
        _finite_values(quantized, tensor)
        rows.append({"name": tensor.name, "dimensions": list(tensor.dimensions), "elements": tensor.elements,
                     "reference_kind": original.kind, "quantized_kind": tensor.kind,
                     "reference_sha256": reference.hash_tensor(original), "quantized_sha256": quantized.hash_tensor(tensor),
                     "reference_bytes": original.nbytes, "quantized_bytes": tensor.nbytes})
    result = validate_plan({"kind": PLAN_KIND, "schema_version": 1, "reference": pair["reference"], "quantized": pair["quantized"],
                           "architecture": "llama", "provenance": declared, "selected": rows, "added_weight_bytes": added,
                           "max_added_bytes": budget, "max_elements": work, "lineage": "operator_declared_compatibility_checked",
                           "behavioral_validation": "not_run"})
    reference.assert_unchanged()
    quantized.assert_unchanged()
    return result


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _layout(reference: GGUFFile, quantized: GGUFFile, selected: set[str]) -> tuple[bytes, list[tuple[TensorInfo, GGUFFile, TensorInfo, int]], int, int]:
    metadata = [entry.encoded for key, entry in quantized.metadata.items() if key != "general.file_type"]
    parts = [struct.pack("<4sIQQ", b"GGUF", 3, len(quantized.tensors), len(metadata)), *metadata]
    layout = []
    position = 0
    for tensor in quantized.tensors:
        source = reference if tensor.name in selected else quantized
        payload = source.tensor(tensor.name)
        position = _align(position, quantized.alignment)
        name = tensor.name.encode("utf-8")
        parts.extend([struct.pack("<Q", len(name)), name, struct.pack("<I", len(tensor.dimensions)),
                      struct.pack("<" + "Q" * len(tensor.dimensions), *tensor.dimensions), struct.pack("<IQ", payload.kind, position)])
        layout.append((tensor, source, payload, position))
        position += payload.nbytes
    header = b"".join(parts)
    data_offset = _align(len(header), quantized.alignment)
    total = data_offset + position
    if total > min(DEFAULT_LIMITS.max_output_bytes, reference.limits.max_output_bytes, quantized.limits.max_output_bytes):
        fail("repair_output_limit", "The rebuilt GGUF would exceed its output-size limit.")
    return header, layout, data_offset, total


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


class _Output:
    """Hold owned directory descriptors; never recursively remove an output tree."""

    def __init__(self, path: str | Path, estimated: int, reserve: int) -> None:
        self.parent_fd: int | None = None
        self.root_fd: int | None = None
        self.root_identity: tuple[int, int] | None = None
        self.owned: dict[str, tuple[int, int]] = {}
        self.completed = False
        if os.name != "posix":
            fail("repair_platform_unsupported", "Create-only repair currently supports POSIX macOS and Linux.")
        try:
            self.path = Path(path).absolute()
            if self.path.name in ("", ".", ".."):
                fail("repair_output_invalid", "Choose a new output directory in an existing parent.")
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            self.parent_fd = os.open(self.path.parent, flags)
            self._space(estimated, reserve, self.parent_fd)
            os.mkdir(self.path.name, 0o700, dir_fd=self.parent_fd)
            self.root_identity = _identity(os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False))
            self.root_fd = os.open(self.path.name, flags, dir_fd=self.parent_fd)
            self.check()
            self._space(estimated, reserve, self.root_fd)
        except FileExistsError:
            self.cleanup()
            self.close()
            fail("repair_output_exists", "The output directory must not already exist, even if empty.")
        except BaseException:
            self.cleanup()
            self.close()
            raise

    @staticmethod
    def _space(estimated: int, reserve: int, descriptor: int) -> None:
        space = os.fstatvfs(descriptor)
        if space.f_bavail * space.f_frsize < estimated + reserve:
            fail("repair_storage_insufficient", "Available storage does not cover the candidate bundle and required reserve.")

    def check(self) -> None:
        if self.parent_fd is None or self.root_fd is None or self.root_identity is None:
            fail("repair_output_changed", "The owned output directory is unavailable.")
        current = os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
        held = os.fstat(self.root_fd)
        if not stat.S_ISDIR(current.st_mode) or _identity(current) != self.root_identity or _identity(held) != self.root_identity:
            fail("repair_output_changed", "The output directory changed during repair.")

    def check_file(self, name: str) -> None:
        self.check()
        current = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode) or _identity(current) != self.owned.get(name):
            fail("repair_output_changed", "An owned output file changed identity during repair.")

    def check_reader(self, name: str, reader: GGUFFile) -> None:
        self.check_file(name)
        if reader.file_identity != self.owned.get(name):
            fail("repair_output_changed", "The verified reader does not hold the owned output file.")

    def new_file(self, name: str) -> BinaryIO:
        self.check()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=self.root_fd)
        try:
            self.owned[name] = _identity(os.fstat(descriptor))
            return os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise

    def bytes(self, name: str, value: bytes) -> None:
        with self.new_file(name) as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        self.verify_bytes(name, value)

    def verify_bytes(self, name: str, expected: bytes) -> None:
        self.check_file(name)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(name, flags, dir_fd=self.root_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or _identity(before) != self.owned.get(name) or before.st_size != len(expected):
                fail("repair_output_changed", "An output record no longer matches its verified bytes.")
            offset = 0
            while offset < len(expected):
                chunk = os.read(descriptor, min(_CHUNK, len(expected) - offset))
                if not chunk or chunk != expected[offset:offset + len(chunk)]:
                    fail("repair_output_changed", "An output record no longer matches its verified bytes.")
                offset += len(chunk)
            after = os.fstat(descriptor)
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (before.st_size, before.st_mtime_ns, before.st_ctime_ns):
                fail("repair_output_changed", "An output record changed while it was checked.")
        finally:
            os.close(descriptor)
        self.check_file(name)

    def install(self, partial: str, final: str) -> None:
        self.check_file(partial)
        os.link(partial, final, src_dir_fd=self.root_fd, dst_dir_fd=self.root_fd, follow_symlinks=False)
        self.owned[final] = self.owned[partial]
        self.check_file(final)
        self.remove(partial)
        os.fsync(self.root_fd)

    def remove(self, name: str) -> None:
        current = os.stat(name, dir_fd=self.root_fd, follow_symlinks=False)
        if stat.S_ISREG(current.st_mode) and _identity(current) == self.owned.get(name):
            os.unlink(name, dir_fd=self.root_fd)
            self.owned.pop(name, None)

    def cleanup(self) -> None:
        if self.root_fd is not None:
            for name in list(self.owned):
                try:
                    self.remove(name)
                except OSError:
                    pass
        if self.parent_fd is not None and self.root_identity is not None:
            try:
                current = os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
                if stat.S_ISDIR(current.st_mode) and _identity(current) == self.root_identity:
                    os.rmdir(self.path.name, dir_fd=self.parent_fd)
            except OSError:
                pass  # Keep unexpected files or replaced directories intact.

    def close(self) -> None:
        for name in ("root_fd", "parent_fd"):
            descriptor = getattr(self, name)
            setattr(self, name, None)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def __enter__(self) -> _Output:
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if not self.completed:
            self.cleanup()
        self.close()
        return False


def _zeros(target: BinaryIO, count: int) -> None:
    while count:
        size = min(count, _CHUNK)
        target.write(b"\0" * size)
        count -= size


def _verify(reference: GGUFFile, quantized: GGUFFile, output: _Output, layout: list,
            data_offset: int, total: int, selected: dict[str, dict]) -> dict:
    output.check_file("candidate.gguf.partial")
    limits = replace(quantized.limits, max_file_bytes=quantized.limits.max_output_bytes)
    with GGUFFile(output.path / "candidate.gguf.partial", limits=limits) as candidate:
        output.check_reader("candidate.gguf.partial", candidate)
        expected_metadata = [key for key in quantized.metadata if key != "general.file_type"]
        if list(candidate.metadata) != expected_metadata or any(candidate.metadata[key].encoded != quantized.metadata[key].encoded for key in expected_metadata):
            fail("repair_verification_failed", "Candidate metadata differs from the prescribed metadata preservation.")
        if candidate.file_size != total or candidate.data_offset != data_offset or candidate.alignment != quantized.alignment or len(candidate.tensors) != len(layout):
            fail("repair_verification_failed", "The candidate container layout does not match its rebuilt header.")
        for actual, (original, source, payload, offset) in zip(candidate.tensors, layout):
            if (actual.name, actual.dimensions, actual.kind, actual.offset, actual.nbytes) != (original.name, original.dimensions, payload.kind, offset, payload.nbytes):
                fail("repair_verification_failed", "The candidate tensor inventory differs from its prescribed layout.")
            expected_hash = selected[actual.name]["reference_sha256"] if actual.name in selected else source.hash_tensor(payload)
            if candidate.hash_tensor(actual) != expected_hash:
                fail("repair_verification_failed", "A candidate tensor payload differs from its selected source.")
        result = {"name": "candidate.gguf", "sha256": candidate.sha256(), "size": candidate.file_size}
        candidate.assert_unchanged()
    output.check_file("candidate.gguf.partial")
    reference.assert_unchanged()
    quantized.assert_unchanged()
    return result


def write_candidate(reference: GGUFFile, quantized: GGUFFile, plan: dict, output_dir: str | Path,
                    *, reserve_bytes: int = 1024**3) -> dict:
    accepted = validate_plan(plan)
    reserve = _integer(reserve_bytes, 0, _MAX_INTEGER)
    regenerated = create_plan(reference, quantized, [row["name"] for row in accepted["selected"]], accepted["provenance"],
                              accepted["max_added_bytes"], max_elements=accepted["max_elements"])
    if regenerated != accepted:
        fail("repair_plan_mismatch", "The supplied plan no longer matches the opened inputs and explicit repair intent.")
    plan_bytes = canonical_json_bytes(accepted)
    selected = {row["name"]: row for row in accepted["selected"]}
    header, layout, data_offset, total = _layout(reference, quantized, set(selected))
    # Include both bounded JSON records and possible temporary/final directory
    # entries. Hard-link installation does not duplicate GGUF payload storage.
    estimated = total + len(plan_bytes) + 2 * 1024**2
    try:
        with _Output(output_dir, estimated, reserve) as output:
            output.bytes("plan.json", plan_bytes)
            with output.new_file("candidate.gguf.partial") as target:
                target.write(header)
                _zeros(target, data_offset - len(header))
                position = 0
                for _original, source, payload, offset in layout:
                    _zeros(target, offset - position)
                    copied = 0
                    for chunk in source.iter_tensor_bytes(payload, chunk_bytes=min(_CHUNK, source.limits.stream_chunk_bytes)):
                        target.write(chunk)
                        copied += len(chunk)
                    if copied != payload.nbytes:
                        fail("repair_verification_failed", "A source tensor did not supply its declared byte count.")
                    position = offset + copied
                if target.tell() != total:
                    fail("repair_verification_failed", "The candidate byte count differs from its planned layout.")
                target.flush()
                os.fsync(target.fileno())
            candidate = _verify(reference, quantized, output, layout, data_offset, total, selected)
            if _binding(reference) != accepted["reference"] or _binding(quantized) != accepted["quantized"]:
                fail("repair_plan_mismatch", "Complete source hashes changed while writing the candidate.")
            reference.assert_unchanged()
            quantized.assert_unchanged()
            output.install("candidate.gguf.partial", "candidate.gguf")
            manifest = {"schema_version": 1, "kind": MANIFEST_KIND,
                        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                        "reference": accepted["reference"], "quantized": accepted["quantized"], "output": candidate,
                        "architecture": "llama", "selected": accepted["selected"], "added_weight_bytes": accepted["added_weight_bytes"],
                        "max_added_bytes": accepted["max_added_bytes"], "max_elements": accepted["max_elements"], "reserve_bytes": reserve,
                        "metadata_omissions": ["general.file_type"] if "general.file_type" in quantized.metadata else [],
                        "provenance": accepted["provenance"], "lineage": accepted["lineage"], "behavioral_validation": "not_run",
                        "verification": {"container_reparsed": True, "metadata_preserved": True, "tensor_payloads_verified": True, "source_hashes_verified": True}}
            # Re-open the installed inode and retain its reader fence through
            # completion, including same-inode size/content changes after link.
            limits = replace(quantized.limits, max_file_bytes=quantized.limits.max_output_bytes)
            with GGUFFile(output.path / "candidate.gguf", limits=limits) as installed:
                output.check_reader("candidate.gguf", installed)
                if installed.sha256() != candidate["sha256"]:
                    fail("repair_verification_failed", "The installed candidate differs from the verified container.")
                manifest_bytes = canonical_json_bytes(manifest)
                output.bytes("repair.json.partial", manifest_bytes)
                output.verify_bytes("plan.json", plan_bytes)
                output.check_file("candidate.gguf")
                reference.assert_unchanged()
                quantized.assert_unchanged()
                installed.assert_unchanged()
                output.install("repair.json.partial", "repair.json")
                output.verify_bytes("repair.json", manifest_bytes)
                installed.assert_unchanged()
                reference.assert_unchanged()
                quantized.assert_unchanged()
            output.completed = True
            return manifest
    except QuantopsyError:
        raise
    except (OSError, ValueError, OverflowError, TypeError):
        fail("repair_output_failed", "The create-only candidate bundle could not be completed; only owned files were eligible for cleanup.")
