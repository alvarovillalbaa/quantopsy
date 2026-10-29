"""Bounded, read-only GGUF v3 model containers and streaming weight decoders."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
from typing import Iterator, Mapping

from .contracts import (
    DEFAULT_LIMITS,
    Limits,
    MetadataEntry,
    TensorInfo,
    TYPE_F16,
    TYPE_F32,
    TYPE_Q4_0,
)
from .errors import fail


_HEADER = struct.Struct("<4sIQQ")
_SCALARS = {
    0: struct.Struct("<B"),
    1: struct.Struct("<b"),
    2: struct.Struct("<H"),
    3: struct.Struct("<h"),
    4: struct.Struct("<I"),
    5: struct.Struct("<i"),
    6: struct.Struct("<f"),
    10: struct.Struct("<Q"),
    11: struct.Struct("<q"),
    12: struct.Struct("<d"),
}
_KEY = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", re.ASCII)


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _check_limits(limits: Limits) -> None:
    if type(limits) is not Limits:
        fail("gguf_invalid_limits", "Use the supported immutable GGUF limits.")
    for name in Limits.__dataclass_fields__:
        value = getattr(limits, name)
        if type(value) is not int or value < 0 or value > 2**63 - 1:
            fail("gguf_invalid_limits", "GGUF limits must be bounded nonnegative integers.")
    if limits.stream_chunk_bytes == 0 or limits.max_dimensions > 4:
        fail("gguf_invalid_limits", "Use a positive stream size and at most four dimensions.")
    if limits.max_tensor_name_bytes > 63 or limits.max_key_bytes > 65_535:
        fail("gguf_invalid_limits", "Names exceed the supported GGUF profile limits.")
    if limits.max_alignment > 4096:
        fail("gguf_invalid_limits", "The supported GGUF alignment limit is at most 4096 bytes.")


class GGUFFile:
    """Hold one unchanged regular input open; never write or execute its bytes.

    The admitted subset has at least one tensor, little-endian version 3,
    F32/F16/Q4_0 payloads, and non-nested scalar/string metadata arrays.
    Tensor offsets remain relative to ``data_offset`` in the public records.
    """

    def __init__(self, path: str | Path, limits: Limits = DEFAULT_LIMITS) -> None:
        self._fd: int | None = None
        _check_limits(limits)
        self.limits = limits
        try:
            self.path = Path(path).absolute()
        except (TypeError, ValueError, OSError):
            fail("gguf_input_invalid", "Choose an explicit regular GGUF input file.")
        self.metadata: Mapping[str, MetadataEntry] = MappingProxyType({})
        self.tensors: tuple[TensorInfo, ...] = ()
        self._by_name: dict[str, TensorInfo] = {}
        self._position = 0
        self._metadata_bytes = 0
        self._metadata_values = 0
        self._record_parts: list[bytes] | None = None
        try:
            if not hasattr(os, "pread"):
                fail("gguf_platform_unsupported", "This reader requires the supported POSIX file-read interface.")
            self._open()
            self._parse()
            self.assert_unchanged()
        except BaseException:
            self._close_quietly()
            raise

    def _open(self) -> None:
        try:
            before = os.stat(self.path, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode):
                fail("gguf_input_invalid", "GGUF inputs must be regular files, without a symlink leaf.")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            self._fd = os.open(self.path, flags)
            opened = os.fstat(self._fd)
        except (OSError, ValueError):
            fail("gguf_input_unavailable", "The GGUF input could not be opened read-only.")
        if not stat.S_ISREG(opened.st_mode):
            fail("gguf_input_invalid", "GGUF inputs must be regular files.")
        self._initial = _fingerprint(opened)
        if _fingerprint(before) != self._initial:
            fail("gguf_source_changed", "The GGUF input changed while it was being opened.")
        self.file_size = opened.st_size
        if self.file_size > self.limits.max_file_bytes:
            fail("gguf_file_limit", "The GGUF input exceeds the configured file-size limit.")

    def _open_fd(self) -> int:
        if self._fd is None:
            fail("gguf_closed", "The GGUF input is closed.")
        return self._fd

    def assert_unchanged(self) -> None:
        descriptor = self._open_fd()
        try:
            held = os.fstat(descriptor)
            current = os.stat(self.path, follow_symlinks=False)
        except (OSError, ValueError):
            fail("gguf_source_changed", "The GGUF input is no longer the opened source.")
        if not stat.S_ISREG(current.st_mode) or _fingerprint(held) != self._initial or _fingerprint(current) != self._initial:
            fail("gguf_source_changed", "The GGUF input changed after it was opened.")

    @property
    def file_identity(self) -> tuple[int, int]:
        """Expose the unchanged held file identity for output verification."""
        self.assert_unchanged()
        return self._initial[0], self._initial[1]

    def _pread(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset > self.file_size or size > self.file_size - offset:
            fail("gguf_truncated", "The GGUF input ends before its declared content.")
        pieces: list[bytes] = []
        remaining = size
        while remaining:
            try:
                piece = os.pread(self._open_fd(), min(remaining, self.limits.stream_chunk_bytes), offset)
            except (OSError, OverflowError, ValueError):
                fail("gguf_read_failed", "The GGUF input could not be read.")
            if not piece:
                fail("gguf_truncated", "The GGUF input ends before its declared content.")
            pieces.append(piece)
            offset += len(piece)
            remaining -= len(piece)
        return b"".join(pieces)

    def _read(self, size: int) -> bytes:
        if self._record_parts is not None:
            if size > self.limits.max_metadata_bytes - self._metadata_bytes:
                fail("gguf_metadata_limit", "GGUF metadata exceeds the configured byte limit.")
            self._metadata_bytes += size
        value = self._pread(self._position, size)
        self._position += size
        if self._record_parts is not None:
            self._record_parts.append(value)
        return value

    def _integer(self, format_: str) -> int:
        record = struct.Struct(format_)
        return record.unpack(self._read(record.size))[0]

    def _string(self, maximum: int) -> str:
        length = self._integer("<Q")
        if length > maximum:
            fail("gguf_string_limit", "A GGUF string exceeds its supported byte limit.")
        try:
            return self._read(length).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            fail("gguf_invalid_utf8", "GGUF strings must contain valid UTF-8.")

    def _value(self, kind: int) -> object:
        self._metadata_values += 1
        if self._metadata_values > self.limits.max_metadata_values:
            fail("gguf_metadata_limit", "GGUF metadata exceeds the decoded-value budget.")
        if kind in _SCALARS:
            scalar = _SCALARS[kind]
            value = scalar.unpack(self._read(scalar.size))[0]
            if isinstance(value, float) and not math.isfinite(value):
                fail("gguf_invalid_metadata", "Metadata floating-point values must be finite in this profile.")
            return value
        if kind == 7:
            value = self._read(1)[0]
            if value not in (0, 1):
                fail("gguf_invalid_metadata", "GGUF boolean metadata must be zero or one.")
            return bool(value)
        if kind == 8:
            return self._string(self.limits.max_string_bytes)
        fail("gguf_metadata_type_unsupported", "This GGUF metadata value type is unsupported.")

    def _parse(self) -> None:
        magic, version, tensor_count, metadata_count = _HEADER.unpack(self._read(_HEADER.size))
        if magic != b"GGUF":
            fail("gguf_invalid_magic", "The input is not a GGUF container.")
        if version != 3:
            fail("gguf_version_unsupported", "Only little-endian GGUF version 3 is supported.")
        if not 1 <= tensor_count <= self.limits.max_tensors:
            fail("gguf_tensor_limit", "Use a model container with a supported nonzero tensor count.")
        if metadata_count > self.limits.max_metadata_entries:
            fail("gguf_metadata_limit", "GGUF metadata has too many entries.")
        metadata: dict[str, MetadataEntry] = {}
        for _ in range(metadata_count):
            self._record_parts = []
            key = self._string(self.limits.max_key_bytes)
            if not key or _KEY.fullmatch(key) is None:
                fail("gguf_invalid_metadata", "GGUF metadata keys must be hierarchical lowercase ASCII names.")
            if key in metadata:
                fail("gguf_duplicate_metadata", "GGUF metadata keys must be unique.")
            kind = self._integer("<I")
            array_kind = None
            if kind == 9:
                self._metadata_values += 1
                if self._metadata_values > self.limits.max_metadata_values:
                    fail("gguf_metadata_limit", "GGUF metadata exceeds the decoded-value budget.")
                array_kind = self._integer("<I")
                if array_kind == 9 or array_kind not in (*_SCALARS, 7, 8):
                    fail("gguf_metadata_type_unsupported", "Nested or unknown GGUF metadata arrays are unsupported.")
                count = self._integer("<Q")
                if count > self.limits.max_array_items or count > self.limits.max_metadata_values - self._metadata_values:
                    fail("gguf_metadata_limit", "A GGUF metadata array exceeds its supported item budget.")
                value = tuple(self._value(array_kind) for _ in range(count))
            else:
                value = self._value(kind)
            encoded = b"".join(self._record_parts)
            self._record_parts = None
            metadata[key] = MetadataEntry(key=key, kind=kind, value=value, encoded=encoded, array_kind=array_kind)
        self.metadata = MappingProxyType(metadata)
        if any(key == "split" or key.startswith("split.") for key in metadata):
            fail("gguf_shards_unsupported", "Split or sharded GGUF containers are unsupported.")
        architecture = metadata.get("general.architecture")
        if architecture is None or architecture.kind != 8 or not isinstance(architecture.value, str) or re.fullmatch(r"[a-z0-9]+", architecture.value, re.ASCII) is None:
            fail("gguf_invalid_metadata", "The model must declare a lowercase ASCII alphanumeric architecture.")
        for key in ("general.file_type", "general.quantization_version"):
            if key in metadata and metadata[key].kind != 4:
                fail("gguf_invalid_metadata", "Standard encoding metadata must use unsigned 32-bit values.")
        alignment = metadata.get("general.alignment")
        if alignment is None:
            self.alignment = 32
        elif alignment.kind == 4 and type(alignment.value) is int:
            self.alignment = alignment.value
        else:
            fail("gguf_invalid_alignment", "GGUF alignment must be an unsigned 32-bit value.")
        if self.alignment <= 0 or self.alignment % 8 or self.alignment > self.limits.max_alignment:
            fail("gguf_invalid_alignment", "GGUF alignment must be a supported positive multiple of eight.")
        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = self._string(self.limits.max_tensor_name_bytes)
            if not name or any(ord(character) < 32 or ord(character) == 127 for character in name):
                fail("gguf_invalid_tensor", "GGUF tensor names must be nonempty text without controls.")
            if name in self._by_name:
                fail("gguf_duplicate_tensor", "GGUF tensor names must be unique.")
            rank = self._integer("<I")
            if not 1 <= rank <= self.limits.max_dimensions:
                fail("gguf_invalid_dimensions", "The tensor rank is outside the supported profile.")
            dimensions = tuple(self._integer("<Q") for _ in range(rank))
            elements = 1
            for dimension in dimensions:
                if dimension == 0 or elements > (self.limits.max_file_bytes * 32 // 18) // dimension:
                    fail("gguf_invalid_dimensions", "The tensor dimensions exceed the bounded container profile.")
                elements *= dimension
            kind = self._integer("<I")
            offset = self._integer("<Q")
            if kind == TYPE_F32:
                nbytes = elements * 4
            elif kind == TYPE_F16:
                nbytes = elements * 2
            elif kind == TYPE_Q4_0:
                if dimensions[0] % 32:
                    fail("gguf_invalid_dimensions", "Q4_0 rows must contain whole 32-value blocks.")
                nbytes = elements // 32 * 18
            else:
                fail("gguf_tensor_type_unsupported", "Only F32, F16 and Q4_0 tensor payloads are supported.")
            if offset % self.alignment:
                fail("gguf_invalid_alignment", "Each tensor offset must respect the GGUF alignment.")
            tensor = TensorInfo(name=name, dimensions=dimensions, kind=kind, offset=offset, nbytes=nbytes, elements=elements)
            tensors.append(tensor)
            self._by_name[name] = tensor
        self.data_offset = self._position + (-self._position % self.alignment)
        if self.data_offset > self.file_size:
            fail("gguf_truncated", "The GGUF tensor data area is missing.")
        previous_end = 0
        for tensor in sorted(tensors, key=lambda item: item.offset):
            if tensor.offset < previous_end:
                fail("gguf_overlapping_tensors", "GGUF tensor payload spans must not overlap.")
            if tensor.offset > self.file_size - self.data_offset or tensor.nbytes > self.file_size - self.data_offset - tensor.offset:
                fail("gguf_truncated", "A declared tensor extends beyond the GGUF input.")
            previous_end = tensor.offset + tensor.nbytes
        if any(tensor.kind == TYPE_Q4_0 for tensor in tensors):
            quantization = metadata.get("general.quantization_version")
            if quantization is None or quantization.kind != 4 or type(quantization.value) is not int or quantization.value != 2:
                fail("gguf_quantization_unsupported", "Q4_0 inputs must declare supported quantization version 2.")
        self.tensors = tuple(tensors)

    def tensor(self, name: str) -> TensorInfo:
        self._open_fd()
        if type(name) is not str or name not in self._by_name:
            fail("gguf_tensor_missing", "The selected tensor is not present in this GGUF input.")
        return self._by_name[name]

    def _owned_tensor(self, tensor: TensorInfo) -> None:
        if not isinstance(tensor, TensorInfo) or self._by_name.get(tensor.name) is not tensor:
            fail("gguf_invalid_tensor", "Use a parsed tensor belonging to this opened GGUF input.")

    def iter_tensor_bytes(self, tensor: TensorInfo, chunk_bytes: int | None = None) -> Iterator[bytes]:
        self._owned_tensor(tensor)
        size = self.limits.stream_chunk_bytes if chunk_bytes is None else chunk_bytes
        if type(size) is not int or not 1 <= size <= self.limits.stream_chunk_bytes:
            fail("gguf_invalid_limits", "Use a positive chunk size within the configured stream limit.")
        self.assert_unchanged()
        offset = self.data_offset + tensor.offset
        remaining = tensor.nbytes
        while remaining:
            self.assert_unchanged()
            data = self._pread(offset, min(size, remaining))
            self.assert_unchanged()
            offset += len(data)
            remaining -= len(data)
            yield data
        self.assert_unchanged()

    def iter_values(self, tensor: TensorInfo) -> Iterator[float]:
        self._owned_tensor(tensor)
        width = 4 if tensor.kind == TYPE_F32 else 2 if tensor.kind == TYPE_F16 else 18
        carry = b""
        for chunk in self.iter_tensor_bytes(tensor):
            data = carry + chunk
            usable = len(data) - len(data) % width
            if tensor.kind in (TYPE_F32, TYPE_F16):
                format_ = "<f" if tensor.kind == TYPE_F32 else "<e"
                for (value,) in struct.iter_unpack(format_, memoryview(data)[:usable]):
                    if not math.isfinite(value):
                        fail("gguf_nonfinite_weight", "Selected weight values must be finite.")
                    yield value
            else:
                for offset in range(0, usable, 18):
                    scale = struct.unpack_from("<e", data, offset)[0]
                    if not math.isfinite(scale):
                        fail("gguf_nonfinite_weight", "Selected Q4_0 scales must be finite.")
                    for index in range(16):
                        yield ((data[offset + 2 + index] & 15) - 8) * scale
                    for index in range(16):
                        yield ((data[offset + 2 + index] >> 4) - 8) * scale
            carry = data[usable:]
        if carry:
            fail("gguf_truncated", "A selected tensor ends inside an encoded value block.")

    def hash_tensor(self, tensor: TensorInfo) -> str:
        digest = hashlib.sha256()
        for data in self.iter_tensor_bytes(tensor):
            digest.update(data)
        return digest.hexdigest()

    def sha256(self) -> str:
        self.assert_unchanged()
        digest = hashlib.sha256()
        offset = 0
        while offset < self.file_size:
            self.assert_unchanged()
            data = self._pread(offset, min(self.limits.stream_chunk_bytes, self.file_size - offset))
            self.assert_unchanged()
            digest.update(data)
            offset += len(data)
        self.assert_unchanged()
        return digest.hexdigest()

    def _close_quietly(self) -> None:
        descriptor, self._fd = self._fd, None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def close(self) -> None:
        descriptor, self._fd = self._fd, None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                fail("gguf_close_failed", "The opened GGUF descriptor could not be closed.")

    def __enter__(self) -> GGUFFile:
        try:
            self.assert_unchanged()
        except BaseException:
            self._close_quietly()
            raise
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if exception_type is None:
            self.close()
        else:
            self._close_quietly()
        return False

    def __del__(self) -> None:
        try:
            self._close_quietly()
        except Exception:
            pass
