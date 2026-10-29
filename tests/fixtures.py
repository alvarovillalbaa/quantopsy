"""Independent GGUF v3 byte fixtures, not model weights or production encoders.

All offsets, scalar encodings and Q4 expected values are authored here from the
container/block contract. No Quantopsy reader, decoder or writer builds them.
Every constructed container is capped at 64 KiB, including malformed fixtures.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct


F32 = 0
F16 = 1
Q4_0 = 2
Q_NAME = "blk.0.attn_q.weight"
UP_NAME = "blk.0.ffn_up.weight"
NORM_NAME = "output_norm.weight"

# A Q4_0 block is a binary16 scale followed by sixteen packed bytes. Low
# nibbles describe positions 0..15; high nibbles describe positions 16..31.
PACKED_NIBBLES = bytes.fromhex("80 91 a2 b3 c4 d5 e6 f7 08 19 2a 3b 4c 5d 6e 7f")
Q4_POSITIVE = bytes.fromhex("00 3c") + PACKED_NIBBLES
Q4_NEGATIVE = bytes.fromhex("00 b8") + PACKED_NIBBLES
Q4_ZERO = bytes.fromhex("00 00") + bytes.fromhex("ff " * 16)
EXPECTED_POSITIVE = (
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
)
EXPECTED_NEGATIVE = (
    4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5,
    0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5,
    0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5,
    4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5,
)


@dataclass(frozen=True)
class Tensor:
    name: str | bytes
    dimensions: tuple[int, ...]
    kind: int
    payload: bytes
    offset: int | None = None


def string(value: str | bytes) -> bytes:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def metadata_value(kind: int, value: object) -> bytes:
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
    if kind in formats:
        return struct.pack("<" + formats[kind], value)
    if kind == 8:
        return string(value)
    if kind == 9:
        element_kind, values = value
        return struct.pack("<IQ", element_kind, len(values)) + b"".join(
            metadata_value(element_kind, element) for element in values)
    # Unknown types need no pretend encoding. The reader must reject their tag.
    return value if isinstance(value, bytes) else b""


def metadata_record(key: str | bytes, kind: int, value: object) -> bytes:
    return string(key) + struct.pack("<I", kind) + metadata_value(kind, value)


def identity_metadata(*, quantized: bool = False, alignment: int = 32):
    metadata = [
        ("general.architecture", 8, "llama"),
        ("general.name", 8, "Tiny authored Llama"),
        ("general.basename", 8, "authored-only"),
        ("general.finetune", 8, "independent-fixture"),
        ("general.size_label", 8, "tiny"),
        ("general.alignment", 4, alignment),
        ("general.file_type", 4, 2 if quantized else 1),
        ("llama.context_length", 4, 64),
        ("llama.embedding_length", 4, 32),
        ("llama.block_count", 4, 1),
        ("tokenizer.ggml.model", 8, "llama"),
        ("tokenizer.ggml.tokens", 9, (8, ["authored-a", "authored-b"])),
        ("tokenizer.ggml.bos_token_id", 4, 0),
    ]
    if quantized:
        metadata.append(("general.quantization_version", 4, 2))
    return metadata


def replace_metadata(metadata, key, kind, value):
    return [(name, tag, data) for name, tag, data in metadata if name != key] + [(key, kind, value)]


def gguf_bytes(tensors=None, metadata=None, *, alignment=32,
               magic=b"GGUF", version=3, tensor_count=None, metadata_count=None):
    if tensors is None:
        tensors = [Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1.0, -1.0))]
    if metadata is None:
        metadata = identity_metadata()
    offsets, cursor = [], 0
    for tensor in tensors:
        offset = tensor.offset if tensor.offset is not None else ((cursor + alignment - 1) // alignment) * alignment
        offsets.append(offset)
        cursor = max(cursor, offset + len(tensor.payload))
    header = magic + struct.pack("<IQQ", version,
                                 len(tensors) if tensor_count is None else tensor_count,
                                 len(metadata) if metadata_count is None else metadata_count)
    header += b"".join(metadata_record(*entry) for entry in metadata)
    for tensor, offset in zip(tensors, offsets):
        header += string(tensor.name) + struct.pack("<I", len(tensor.dimensions))
        header += b"".join(struct.pack("<Q", dimension) for dimension in tensor.dimensions)
        header += struct.pack("<IQ", tensor.kind, offset)
    data_offset = ((len(header) + alignment - 1) // alignment) * alignment
    if data_offset + cursor > 64 * 1024:
        raise ValueError("Authored fixture exceeds 64 KiB")
    output = bytearray(data_offset + cursor)
    output[:len(header)] = header
    for tensor, offset in zip(tensors, offsets):
        output[data_offset + offset:data_offset + offset + len(tensor.payload)] = tensor.payload
    return bytes(output)


def reference_tensors():
    return [
        Tensor(Q_NAME, (32, 2), F16, struct.pack("<64e", *[index / 4 for index in range(64)])),
        Tensor(UP_NAME, (32, 1), F16, struct.pack("<32e", *([1.25] * 32))),
        Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1.0, -1.0)),
    ]


def quantized_tensors():
    return [Tensor(Q_NAME, (32, 2), Q4_0, Q4_POSITIVE + Q4_NEGATIVE),
            Tensor(UP_NAME, (32, 1), Q4_0, Q4_ZERO),
            Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1.0, -1.0))]


@dataclass(frozen=True)
class Pair:
    reference: Path
    quantized: Path
    reference_bytes: bytes
    quantized_bytes: bytes


def write_pair(directory: Path, *, reference=None, quantized=None,
               reference_meta=None, quantized_meta=None, alignment=32) -> Pair:
    ref = gguf_bytes(reference if reference is not None else reference_tensors(),
                     reference_meta if reference_meta is not None else identity_metadata(alignment=alignment), alignment=alignment)
    quant = gguf_bytes(quantized if quantized is not None else quantized_tensors(),
                       quantized_meta if quantized_meta is not None else identity_metadata(quantized=True, alignment=alignment), alignment=alignment)
    reference_path, quantized_path = directory / "reference.gguf", directory / "quantized.gguf"
    reference_path.write_bytes(ref)
    quantized_path.write_bytes(quant)
    return Pair(reference_path, quantized_path, ref, quant)


def provenance():
    return {"model_id": "authored/tiny-llama", "revision": "authored-frozen-r1",
            "license": "authored fixture only", "reference_source": "authored reference bytes",
            "quantized_source": "authored quantized bytes"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
