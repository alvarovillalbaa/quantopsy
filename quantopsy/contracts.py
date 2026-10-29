"""Shared immutable limits and parsed GGUF records."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Limits:
    max_file_bytes: int = 8 * 1024**3
    max_output_bytes: int = 16 * 1024**3
    max_metadata_bytes: int = 32 * 1024**2
    max_metadata_entries: int = 4096
    max_tensors: int = 8192
    max_array_items: int = 262144
    max_metadata_values: int = 1048576
    max_string_bytes: int = 1024**2
    max_key_bytes: int = 1024
    max_tensor_name_bytes: int = 63
    max_dimensions: int = 4
    max_alignment: int = 4096
    stream_chunk_bytes: int = 1024**2
    max_selected_elements: int = 64_000_000


DEFAULT_LIMITS = Limits()
TYPE_F32 = 0
TYPE_F16 = 1
TYPE_Q4_0 = 2
TYPE_NAMES = {TYPE_F32: "F32", TYPE_F16: "F16", TYPE_Q4_0: "Q4_0"}


@dataclass(frozen=True, slots=True)
class MetadataEntry:
    key: str
    kind: int
    value: object
    encoded: bytes
    array_kind: int | None = None


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    kind: int
    offset: int
    nbytes: int
    elements: int
