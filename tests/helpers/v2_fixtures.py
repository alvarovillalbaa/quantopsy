"""Tier 2 independent authored v2 GGUF bytes, capped by tests.fixtures at 64 KiB.

No application reader, encoder or writer constructs these fixture containers.
The Q8 scales deliberately include a nonfinite bit pattern: the v2 contract
copies this unselected payload opaquely and makes no numerical claim about it.
"""
from pathlib import Path
import struct

from tests.fixtures import (
    F16, F32, NORM_NAME, Q4_0, Q4_NEGATIVE, Q4_POSITIVE, Q4_ZERO,
    Q_NAME, UP_NAME, Tensor, gguf_bytes, identity_metadata, replace_metadata,
    write_pair,
)

Q8_0 = 8
OPAQUE_NAME = "output.weight"
F16_NAME = "blk.0.ffn_norm.weight"
F32_SELECTED = struct.pack("<64f", *[index / 8 for index in range(64)])
Q8_BLOCK = bytes.fromhex("00 7e") + struct.pack("<32b", *range(-16, 16))
Q8_SECOND = bytes.fromhex("00 3c") + struct.pack("<32b", *range(15, -17, -1))
OPAQUE_Q8 = Q8_BLOCK + Q8_SECOND


def reference_tensors():
    return [
        Tensor(Q_NAME, (32, 2), F32, F32_SELECTED),
        Tensor(UP_NAME, (32, 1), F32, struct.pack("<32f", *[index / 16 for index in range(32)])),
        Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1.0, -1.0)),
        Tensor(F16_NAME, (2,), F16, struct.pack("<2e", 0.5, -0.5)),
        Tensor(OPAQUE_NAME, (32, 2), F32, struct.pack("<64f", *[index / 32 for index in range(64)])),
    ]


def quantized_tensors():
    return [
        Tensor(Q_NAME, (32, 2), Q4_0, Q4_POSITIVE + Q4_NEGATIVE),
        Tensor(UP_NAME, (32, 1), Q4_0, Q4_ZERO),
        Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1.0, -1.0)),
        Tensor(F16_NAME, (2,), F16, struct.pack("<2e", 0.5, -0.5)),
        Tensor(OPAQUE_NAME, (32, 2), Q8_0, OPAQUE_Q8),
    ]


def reference_metadata():
    return replace_metadata(identity_metadata(), "general.file_type", 4, 0)


def quantized_metadata():
    return identity_metadata(quantized=True)


def write_v2_pair(directory: Path, *, reference=None, quantized=None,
                  reference_meta=None, quantized_meta=None):
    return write_pair(
        directory,
        reference=reference_tensors() if reference is None else reference,
        quantized=quantized_tensors() if quantized is None else quantized,
        reference_meta=reference_metadata() if reference_meta is None else reference_meta,
        quantized_meta=quantized_metadata() if quantized_meta is None else quantized_meta,
    )


def q8_container(*, dimensions=(32, 2), payload=OPAQUE_Q8, metadata=None):
    return gguf_bytes(
        [Tensor(OPAQUE_NAME, dimensions, Q8_0, payload)],
        quantized_metadata() if metadata is None else metadata,
    )


def expected_candidate_bytes(selected=(Q_NAME,)):
    originals = {tensor.name: tensor for tensor in reference_tensors()}
    prescribed = [originals[tensor.name] if tensor.name in selected else tensor
                  for tensor in quantized_tensors()]
    metadata = [entry for entry in quantized_metadata() if entry[0] != "general.file_type"]
    return gguf_bytes(prescribed, metadata)
