"""Twenty-six authored reader/decoder cases. No model or inference fixture."""

from dataclasses import replace
import math
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from quantopsy.contracts import Limits
from quantopsy.errors import QuantopsyError
from quantopsy.gguf import GGUFFile
from tests.fixtures import (
    EXPECTED_NEGATIVE, EXPECTED_POSITIVE, F16, F32, NORM_NAME,
    Q4_0, Q4_NEGATIVE, Q4_POSITIVE, Q4_ZERO, Q_NAME, Tensor,
    gguf_bytes, identity_metadata, metadata_record, quantized_tensors,
    replace_metadata, sha256, string, write_pair,
)


class GGUFTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="quantopsy-authored-reader-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, data, name="input.gguf"):
        path = self.root / name
        path.write_bytes(data)
        return path

    def rejected(self, data, limits=Limits()):
        with self.assertRaises(QuantopsyError) as caught:
            with GGUFFile(self.write(data), limits=limits):
                pass
        self.assertTrue(caught.exception.code.startswith("gguf_"))

    def test_v3_inventory_identity_and_original_metadata_bytes(self):
        pair = write_pair(self.root)
        with GGUFFile(pair.quantized) as model:
            self.assertEqual(model.file_size, len(pair.quantized_bytes))
            self.assertEqual(model.alignment, 32)
            self.assertEqual(model.data_offset % 32, 0)
            self.assertEqual([tensor.name for tensor in model.tensors], [tensor.name for tensor in quantized_tensors()])
            tensor = model.tensor(Q_NAME)
            self.assertEqual((tensor.dimensions, tensor.kind, tensor.nbytes, tensor.elements), ((32, 2), 2, 36, 64))
            for key, kind, value in identity_metadata(quantized=True):
                expected = metadata_record(key, kind, value)
                self.assertEqual(model.metadata[key].encoded, expected)
                self.assertEqual(sha256(model.metadata[key].encoded), sha256(expected))
        with self.assertRaises(QuantopsyError):
            model.sha256()

    def test_scalar_and_flat_array_metadata_have_independent_encodings(self):
        entries = [("general.architecture", 8, "llama"),
                   ("fixture.u8", 0, 255), ("fixture.i8", 1, -12),
                   ("fixture.u16", 2, 65000), ("fixture.i16", 3, -32000),
                   ("fixture.u32", 4, 2**32 - 1), ("fixture.i32", 5, -123456),
                   ("fixture.f32", 6, 1.25), ("fixture.boolean", 7, True),
                   ("fixture.text", 8, "café 界"), ("fixture.u64", 10, 2**63 + 3),
                   ("fixture.i64", 11, -(2**50)), ("fixture.f64", 12, -0.125),
                   ("fixture.strings", 9, (8, ["one", "界"])),
                   ("fixture.numbers", 9, (5, [-2, 0, 3]))]
        with GGUFFile(self.write(gguf_bytes(metadata=entries))) as model:
            for key, kind, value in entries:
                expected_value = tuple(value[1]) if kind == 9 else value
                self.assertEqual(model.metadata[key].value, expected_value)
                self.assertEqual(model.metadata[key].encoded, metadata_record(key, kind, value))
            self.assertEqual(model.metadata["fixture.strings"].array_kind, 8)
            with self.assertRaises(TypeError):
                model.metadata["invented"] = model.metadata["fixture.text"]

    def test_f32_values_preserve_sign_fraction_and_negative_zero(self):
        values = [1.0, -2.5, 0.0, -0.0, 3.4028234663852886e38]
        tensor = Tensor("fixture.weight", (5,), F32, struct.pack("<5f", *values))
        with GGUFFile(self.write(gguf_bytes([tensor]))) as model:
            actual = list(model.iter_values(model.tensor(tensor.name)))
            self.assertEqual(actual, values)
            self.assertEqual(math.copysign(1, actual[3]), -1)

    def test_f16_values_preserve_independently_packed_binary16_edges(self):
        values = [1.0, -0.5, 65504.0, 2**-24, -0.0]
        tensor = Tensor("fixture.weight", (5,), F16, struct.pack("<5e", *values))
        with GGUFFile(self.write(gguf_bytes([tensor]))) as model:
            actual = list(model.iter_values(model.tensor(tensor.name)))
            self.assertEqual(actual, values)
            self.assertEqual(math.copysign(1, actual[-1]), -1)

    def test_q4_low_nibbles_precede_high_nibbles_for_one_literal_block(self):
        tensor = Tensor(Q_NAME, (32, 1), Q4_0, Q4_POSITIVE)
        with GGUFFile(self.write(gguf_bytes([tensor], identity_metadata(quantized=True)))) as model:
            self.assertEqual(tuple(model.iter_values(model.tensor(Q_NAME))), EXPECTED_POSITIVE)

    def test_q4_negative_and_zero_scales_preserve_rows_without_positive_fabrication(self):
        tensor = Tensor(Q_NAME, (32, 2), Q4_0, Q4_NEGATIVE + Q4_ZERO)
        with GGUFFile(self.write(gguf_bytes([tensor], identity_metadata(quantized=True)))) as model:
            actual = tuple(model.iter_values(model.tensor(Q_NAME)))
            self.assertEqual(actual[:32], EXPECTED_NEGATIVE)
            self.assertEqual(actual[32:], (0.0,) * 32)

    def test_one_byte_streams_and_hashes_cover_exact_payloads_and_fresh_file(self):
        pair = write_pair(self.root)
        with GGUFFile(pair.quantized, limits=Limits(stream_chunk_bytes=1)) as model:
            self.assertEqual(model.sha256(), sha256(pair.quantized_bytes))
            for expected in quantized_tensors():
                tensor = model.tensor(expected.name)
                self.assertEqual(b"".join(model.iter_tensor_bytes(tensor)), expected.payload)
                self.assertEqual(model.hash_tensor(tensor), sha256(expected.payload))
            self.assertEqual(tuple(model.iter_values(model.tensor(Q_NAME))), EXPECTED_POSITIVE + EXPECTED_NEGATIVE)
            with self.assertRaises(QuantopsyError):
                model.hash_tensor(replace(model.tensor(Q_NAME)))
            with self.assertRaises(QuantopsyError):
                list(model.iter_tensor_bytes(model.tensor(Q_NAME), chunk_bytes=0))

    def test_magic_endianness_and_unsupported_versions_are_not_guessed(self):
        valid = gguf_bytes()
        for changed in [b"FUGG" + valid[4:], valid[:4] + struct.pack(">I", 3) + valid[8:],
                        valid[:4] + struct.pack("<I", 2) + valid[8:],
                        valid[:4] + struct.pack("<I", 4) + valid[8:]]:
            with self.subTest(prefix=changed[:8]): self.rejected(changed)

    def test_unsupported_tensor_encodings_do_not_receive_guessed_sizes(self):
        for kind in [3, 8, 14, 99]:
            with self.subTest(kind=kind):
                self.rejected(gguf_bytes([Tensor("fixture.weight", (32, 1), kind, b"\0" * 32)]))

    def test_shard_metadata_and_tokenizer_only_zero_tensor_files_are_rejected(self):
        for key, kind, value in [("split.count", 2, 2), ("split.no", 2, 0), ("split.tensors.count", 5, 3)]:
            with self.subTest(key=key): self.rejected(gguf_bytes(metadata=identity_metadata() + [(key, kind, value)]))
        self.rejected(gguf_bytes(tensors=[]))

    def test_invalid_utf8_in_keys_values_and_tensor_names_is_not_replaced(self):
        for data in [gguf_bytes(metadata=identity_metadata() + [(b"\xff", 8, "value")]),
                     gguf_bytes(metadata=identity_metadata() + [("fixture.text", 8, b"\xff")]),
                     gguf_bytes([Tensor(b"\xff", (1,), F32, b"\0" * 4)])]:
            self.rejected(data)

    def test_duplicate_metadata_and_tensor_names_are_refused(self):
        self.rejected(gguf_bytes(metadata=identity_metadata() + [("general.name", 8, "other")]))
        tensor = Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1.0, -1.0))
        self.rejected(gguf_bytes([tensor, tensor]))

    def test_truncated_headers_and_declared_strings_fail_without_large_buffers(self):
        valid = gguf_bytes()
        for size in [0, 4, 8, 16, 23]:
            with self.subTest(size=size): self.rejected(valid[:size])
        self.rejected(b"GGUF" + struct.pack("<IQQQ", 3, 1, 1, 2**40))

    def test_unknown_nested_boolean_and_nonfinite_metadata_types_are_rejected(self):
        entries = [("fixture.unknown", 99, b""), ("fixture.nested", 9, (9, [(8, ["nested"])])),
                   ("fixture.boolean", 7, 2), ("fixture.float", 6, float("nan")),
                   ("fixture.double", 12, float("inf"))]
        for entry in entries:
            with self.subTest(key=entry[0]): self.rejected(gguf_bytes(metadata=identity_metadata() + [entry]))
        for kind, architecture in [(8, ""), (8, "Llama"), (8, "future-llama"),
                                   (8, "llama\0"), (4, 1)]:
            with self.subTest(architecture=architecture):
                self.rejected(gguf_bytes(metadata=replace_metadata(identity_metadata(), "general.architecture", kind, architecture)))
        self.rejected(gguf_bytes(metadata=[entry for entry in identity_metadata() if entry[0] != "general.architecture"]))

    def test_declared_metadata_and_tensor_counts_are_bounded_before_allocation(self):
        self.rejected(b"GGUF" + struct.pack("<IQQ", 3, 8193, 0))
        self.rejected(b"GGUF" + struct.pack("<IQQ", 3, 1, 4097))
        declared_array = string("fixture.array") + struct.pack("<IIQ", 9, 8, 2**40)
        self.rejected(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + declared_array)

    def test_zero_excessive_rank_and_overflowing_dimensions_are_rejected(self):
        for shape in [(), (0,), (1, 1, 1, 1, 1), (2**63, 32)]:
            with self.subTest(shape=shape): self.rejected(gguf_bytes([Tensor("fixture.weight", shape, F32, b"\0" * 4)]))

    def test_q4_rows_and_quantization_version_must_match_the_supported_profile(self):
        for shape in [(31, 1), (16, 2)]:
            self.rejected(gguf_bytes([Tensor(Q_NAME, shape, Q4_0, Q4_POSITIVE)], identity_metadata(quantized=True)))
        for kind, version in [(4, 1), (6, 2.0)]:
            metadata = replace_metadata(identity_metadata(quantized=True), "general.quantization_version", kind, version)
            self.rejected(gguf_bytes([Tensor(Q_NAME, (32, 1), Q4_0, Q4_POSITIVE)], metadata))
        metadata = [entry for entry in identity_metadata(quantized=True) if entry[0] != "general.quantization_version"]
        self.rejected(gguf_bytes([Tensor(Q_NAME, (32, 1), Q4_0, Q4_POSITIVE)], metadata))

    def test_alignment_accepts_multiples_of_eight_and_rejects_invalid_metadata(self):
        for alignment in [8, 24, 64]:
            with GGUFFile(self.write(gguf_bytes(metadata=identity_metadata(alignment=alignment), alignment=alignment))) as model:
                self.assertEqual(model.alignment, alignment)
                self.assertEqual(model.data_offset % alignment, 0)
        metadata = [("general.architecture", 8, "llama")]
        data = bytearray(gguf_bytes(metadata=metadata))
        header_end = 24 + sum(len(metadata_record(*entry)) for entry in metadata)
        header_end += len(string(NORM_NAME)) + 4 + 8 + 4 + 8
        data_offset = len(data) - 8
        self.assertLess(header_end, data_offset)
        data[header_end:data_offset] = b"\xa5" * (data_offset - header_end)
        with GGUFFile(self.write(bytes(data))) as model:
            self.assertEqual(tuple(model.iter_values(model.tensor(NORM_NAME))), (1.0, -1.0))
            self.assertEqual(model.data_offset, data_offset)
        for kind, alignment in [(4, 0), (4, 7), (4, 12), (4, 4104), (8, "32")]:
            self.rejected(gguf_bytes(metadata=replace_metadata(identity_metadata(), "general.alignment", kind, alignment)))

    def test_tensor_relative_offsets_must_be_aligned(self):
        self.rejected(gguf_bytes([Tensor(NORM_NAME, (2,), F32, b"\0" * 8, offset=1)]))

    def test_overlapping_payload_spans_are_rejected_independent_of_table_order(self):
        first = Tensor("one.weight", (8,), F32, b"\0" * 32, offset=0)
        second = Tensor("two.weight", (2,), F32, b"\0" * 8, offset=0)
        for tensors in [[first, second], [second, first]]: self.rejected(gguf_bytes(tensors))

    def test_payload_truncation_cannot_be_hidden_by_valid_header_and_offsets(self):
        pair = write_pair(self.root)
        self.rejected(pair.quantized_bytes[:-1])
        self.rejected(pair.reference_bytes[:-1])

    def test_file_string_metadata_array_and_global_value_budgets_are_explicit(self):
        data = gguf_bytes()
        with GGUFFile(self.write(data), limits=Limits(max_file_bytes=len(data))) as model:
            self.assertEqual(model.file_size, len(data))
        self.rejected(data, Limits(max_file_bytes=len(data) - 1))
        for limits in [Limits(max_metadata_bytes=12), Limits(max_metadata_entries=1),
                       Limits(max_array_items=1), Limits(max_metadata_values=3),
                       Limits(max_key_bytes=4), Limits(max_string_bytes=4), Limits(max_tensor_name_bytes=3)]:
            with self.subTest(limits=limits): self.rejected(data, limits)
        for limits in [Limits(stream_chunk_bytes=0), Limits(max_dimensions=5), Limits(max_tensors=True)]:
            with self.assertRaises(QuantopsyError): GGUFFile(self.write(data), limits=limits)

    def test_source_mutation_and_path_replacement_invalidate_open_streams_and_hashes(self):
        for replacement in [False, True]:
            path = self.write(gguf_bytes())
            with GGUFFile(path) as model:
                self.assertEqual(model.sha256(), sha256(path.read_bytes()))
                if replacement:
                    other = self.write(gguf_bytes(), "replacement.gguf"); os.replace(other, path)
                else:
                    old = path.stat(); data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
                    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
                for action in [model.assert_unchanged, model.sha256, lambda: model.hash_tensor(model.tensor(NORM_NAME)),
                               lambda: list(model.iter_values(model.tensor(NORM_NAME)))]:
                    with self.assertRaises(QuantopsyError) as caught: action()
                    self.assertEqual(caught.exception.code, "gguf_source_changed")

    def test_symlink_leaf_and_directories_are_never_read_as_model_inputs(self):
        path = self.write(gguf_bytes()); link = self.root / "linked.gguf"; link.symlink_to(path)
        for rejected in [link, self.root]:
            with self.assertRaises(QuantopsyError): GGUFFile(rejected)
        self.assertEqual(path.read_bytes(), gguf_bytes())

    def test_fifo_refusal_and_nonblocking_nofollow_descriptor_flags_are_explicit(self):
        fifo = self.root / "pipe.gguf"; os.mkfifo(fifo)
        path = self.write(gguf_bytes()); original_open = os.open; flags_seen = []
        def guarded_open(file, flags, *args, **kwargs):
            flags_seen.append(flags)
            # Keep this negative fixture bounded even if a regression drops the flag.
            return original_open(file, flags | os.O_NONBLOCK, *args, **kwargs)
        with patch("quantopsy.gguf.os.open", side_effect=guarded_open):
            with self.assertRaises(QuantopsyError): GGUFFile(fifo)
            with GGUFFile(path): pass
        self.assertTrue(flags_seen)
        self.assertTrue(all(flags & os.O_NONBLOCK for flags in flags_seen))
        if hasattr(os, "O_NOFOLLOW"): self.assertTrue(all(flags & os.O_NOFOLLOW for flags in flags_seen))

    def test_nonfinite_f32_f16_and_q4_scale_values_fail_during_decode(self):
        for kind, payload, dimensions in [
            (F32, struct.pack("<f", float("nan")), (1,)),
            (F32, struct.pack("<f", float("inf")), (1,)),
            (F16, struct.pack("<e", -float("inf")), (1,)),
            (Q4_0, bytes.fromhex("00 7c") + b"\0" * 16, (32, 1)),
            (Q4_0, bytes.fromhex("00 7e") + b"\0" * 16, (32, 1)),
        ]:
            metadata = identity_metadata(quantized=kind == Q4_0)
            with self.subTest(kind=kind, payload=payload[:4]):
                with GGUFFile(self.write(gguf_bytes([Tensor("fixture.weight", dimensions, kind, payload)], metadata))) as model:
                    with self.assertRaises(QuantopsyError) as caught: list(model.iter_values(model.tensor("fixture.weight")))
                    self.assertEqual(caught.exception.code, "gguf_nonfinite_weight")
