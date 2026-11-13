"""File/CLI integration, Tier 2: tiny authored GGUF inputs only; UNRUN locally.

No public weights, model engine, provider, subprocess or network fixture is used.
The future owned CI selects this class independently from the historical suite.
"""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from quantopsy.cli import main
from quantopsy.contracts import Limits
from quantopsy.errors import QuantopsyError
from quantopsy.gguf import GGUFFile
from quantopsy.io import canonical_json_bytes
from quantopsy.repair import create_plan, pair_inputs, validate_plan, write_candidate
from tests.fixtures import (
    F16, F32, NORM_NAME, Q4_0, Q4_NEGATIVE, Q4_POSITIVE, Q_NAME, UP_NAME,
    Tensor, gguf_bytes, identity_metadata, provenance, replace_metadata, sha256,
    write_pair,
)
from tests.helpers.v2_fixtures import (
    F16_NAME, F32_SELECTED, OPAQUE_NAME, OPAQUE_Q8, Q8_0,
    expected_candidate_bytes, q8_container, quantized_metadata,
    quantized_tensors, reference_metadata, reference_tensors, write_v2_pair,
)


class ProfileV2IntegrationTier2Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="quantopsy-v2-authored-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.counter = 0

    def folder(self, label="fixture"):
        self.counter += 1
        path = self.root / (label + "-" + str(self.counter))
        path.mkdir()
        return path

    def raw_file(self, data):
        self.counter += 1
        path = self.root / ("authored-" + str(self.counter) + ".gguf")
        path.write_bytes(data)
        return path

    @contextmanager
    def models(self, *, profile="v2", reference_profile=None, quantized_profile=None,
               limits=None, **fixture):
        pair = write_v2_pair(self.folder(), **fixture)
        with ExitStack() as stack:
            options = {} if limits is None else {"limits": limits}
            reference = stack.enter_context(GGUFFile(pair.reference, profile=reference_profile or profile, **options))
            quantized = stack.enter_context(GGUFFile(pair.quantized, profile=quantized_profile or profile, **options))
            yield pair, reference, quantized

    def plan(self, reference, quantized, *, names=None, added=4096, elements=96):
        return create_plan(reference, quantized, [Q_NAME] if names is None else names,
                           provenance(), added, max_elements=elements, profile="v2")

    def reject(self, code, function, *args, **kwargs):
        with self.assertRaises(QuantopsyError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def command(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(list(map(str, arguments)))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_q8_span_is_two_explicit_34_byte_blocks_and_profile_is_read_only(self):
        path = self.raw_file(q8_container())
        with GGUFFile(path, profile="v2") as model:
            tensor = model.tensor(OPAQUE_NAME)
            self.assertEqual((tensor.kind, tensor.dimensions, tensor.elements, tensor.nbytes),
                             (8, (32, 2), 64, 68))
            self.assertEqual(b"".join(model.iter_tensor_bytes(tensor)), OPAQUE_Q8)
            self.assertEqual(model.hash_tensor(tensor), sha256(OPAQUE_Q8))
            self.assertEqual(model.profile, "v2")
            with self.assertRaises(AttributeError):
                model.profile = "v1"

    def test_q8_requires_whole_dimension_zero_rows_not_just_total_blocks(self):
        for dimensions in ((16, 2), (31, 2), (33, 1)):
            with self.subTest(dimensions=dimensions):
                self.reject("gguf_invalid_dimensions", GGUFFile,
                            self.raw_file(q8_container(dimensions=dimensions)), profile="v2")

    def test_q8_truncation_overlap_alignment_and_unknown_encodings_are_refused(self):
        metadata = quantized_metadata()
        samples = [
            ("gguf_truncated", q8_container()[:-1]),
            ("gguf_overlapping_tensors", gguf_bytes([
                Tensor(OPAQUE_NAME, (32, 2), Q8_0, OPAQUE_Q8, 0),
                Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1, -1), 32)], metadata)),
            ("gguf_invalid_alignment", gguf_bytes([
                Tensor(OPAQUE_NAME, (32, 2), Q8_0, OPAQUE_Q8, 8)], metadata)),
            ("gguf_tensor_type_unsupported", gguf_bytes([
                Tensor(OPAQUE_NAME, (32, 2), 9, OPAQUE_Q8)], metadata)),
        ]
        for code, data in samples:
            with self.subTest(code=code):
                self.reject(code, GGUFFile, self.raw_file(data), profile="v2")

    def test_q8_without_q4_still_requires_exact_quantization_version_two(self):
        for version in (None, 0, 1, 3):
            metadata = [row for row in quantized_metadata() if row[0] != "general.quantization_version"]
            if version is not None:
                metadata.append(("general.quantization_version", 4, version))
            with self.subTest(version=version):
                self.reject("gguf_quantization_unsupported", GGUFFile,
                            self.raw_file(q8_container(metadata=metadata)), profile="v2")

    def test_default_v1_refuses_q8_and_keeps_explicit_v1_reports_and_candidate_bytes(self):
        self.reject("gguf_tensor_type_unsupported", GGUFFile, self.raw_file(q8_container()))
        pair = write_pair(self.folder("v1"))
        default = self.command("inspect", pair.quantized)
        explicit = self.command("inspect", pair.quantized, "--profile", "v1")
        self.assertEqual(default, explicit)
        self.assertEqual(default[0], 0)
        self.assertEqual(json.loads(default[1])["schema_version"], 1)
        with GGUFFile(pair.reference) as reference, GGUFFile(pair.quantized) as quantized:
            first = create_plan(reference, quantized, [Q_NAME], provenance(), 4096)
            second = create_plan(reference, quantized, [Q_NAME], provenance(), 4096, profile="v1")
            self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
            a, b = self.root / "v1-default", self.root / "v1-explicit"
            ma = write_candidate(reference, quantized, first, a, reserve_bytes=0)
            mb = write_candidate(reference, quantized, second, b, reserve_bytes=0, profile="v1")
            self.assertEqual(ma, mb)
            self.assertEqual((a / "candidate.gguf").read_bytes(), (b / "candidate.gguf").read_bytes())

    def test_opaque_q8_decoder_refuses_before_any_tensor_stream_is_consumed(self):
        with GGUFFile(self.raw_file(q8_container()), profile="v2") as model:
            with patch.object(GGUFFile, "iter_tensor_bytes", side_effect=AssertionError("Q8 is opaque")) as stream:
                self.reject("gguf_tensor_decode_unsupported", lambda: list(model.iter_values(model.tensor(OPAQUE_NAME))))
            stream.assert_not_called()

    def test_invalid_profile_values_are_not_coerced_and_fail_before_input_open(self):
        for value in (None, True, False, 1, 2, 2.0, "", "V2", "v4"):
            with self.subTest(profile=value):
                with patch("quantopsy.gguf.os.open", side_effect=AssertionError("profile must be validated first")) as opened:
                    self.reject("gguf_invalid_profile", GGUFFile, self.root / "missing.gguf", profile=value)
                opened.assert_not_called()
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            for value in (True, 2, None, "v4"):
                with self.subTest(operation_profile=value):
                    self.reject("gguf_invalid_profile", pair_inputs, reference, quantized, profile=value)
                    self.reject("gguf_invalid_profile", create_plan, reference, quantized, [Q_NAME], provenance(), 4096, profile=value)
                    self.reject("gguf_invalid_profile", validate_plan, plan, profile=value)
                    output = self.root / "invalid-profile-output"
                    self.reject("gguf_invalid_profile", write_candidate, reference, quantized, plan, output, reserve_bytes=0, profile=value)
                    self.assertFalse(output.exists())

    def test_both_readers_and_operation_must_agree_on_the_explicit_profile(self):
        ref = [row for row in reference_tensors() if row.name != OPAQUE_NAME]
        quant = [row for row in quantized_tensors() if row.name != OPAQUE_NAME]
        with self.models(reference=ref, quantized=quant) as (pair, reference, quantized):
            plan = self.plan(reference, quantized)
            with GGUFFile(pair.reference) as wrong_reference:
                self.reject("repair_profile_mismatch", pair_inputs, wrong_reference, quantized, profile="v2")
                self.reject("repair_profile_mismatch", create_plan, wrong_reference, quantized, [Q_NAME], provenance(), 4096, profile="v2")
                output = self.root / "mixed-readers"
                self.reject("repair_profile_mismatch", write_candidate, wrong_reference, quantized, plan, output, reserve_bytes=0, profile="v2")
                self.assertFalse(output.exists())
            with GGUFFile(pair.quantized) as wrong_quantized:
                self.reject("repair_profile_mismatch", pair_inputs, reference, wrong_quantized, profile="v2")
            self.reject("repair_profile_mismatch", pair_inputs, reference, quantized)

    def test_v2_plan_binds_selected_f32_bytes_and_returns_detached_strict_records(self):
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized)
            self.assertEqual(plan["schema_version"], 2)
            row = plan["selected"][0]
            self.assertEqual((row["name"], row["dimensions"], row["elements"]), (Q_NAME, [32, 2], 64))
            self.assertEqual((row["reference_kind"], row["quantized_kind"]), (0, 2))
            self.assertEqual((row["reference_bytes"], row["quantized_bytes"], plan["added_weight_bytes"]), (256, 36, 220))
            self.assertEqual(row["reference_sha256"], sha256(F32_SELECTED))
            self.assertEqual(plan["reference"]["sha256"], sha256(pair.reference_bytes))
            self.assertEqual(plan["quantized"]["sha256"], sha256(pair.quantized_bytes))
            accepted = validate_plan(plan, profile="v2")
            accepted["selected"][0]["dimensions"][0] = 1
            accepted["provenance"]["revision"] = "detached"
            self.assertEqual(plan["selected"][0]["dimensions"], [32, 2])
            self.assertNotEqual(plan["provenance"]["revision"], "detached")

    def test_v2_plan_rejects_boolean_integer_coercions_extra_fields_and_f16_widths(self):
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
        variants = []
        for field, value in (("schema_version", True), ("schema_version", 2.0), ("max_elements", True)):
            bad = deepcopy(plan); bad[field] = value; variants.append(bad)
        for field, value in (("reference_kind", False), ("reference_kind", 0.0),
                             ("elements", 64.0), ("reference_bytes", 128)):
            bad = deepcopy(plan); bad["selected"][0][field] = value; variants.append(bad)
        bad = deepcopy(plan); bad["extra"] = "not supported"; variants.append(bad)
        bad = deepcopy(plan); del bad["selected"]; variants.append(bad)
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(QuantopsyError):
                    validate_plan(value, profile="v2")

    def test_schema_versions_cannot_silently_choose_or_upgrade_the_requested_profile(self):
        pair = write_pair(self.folder("legacy"))
        with GGUFFile(pair.reference) as reference, GGUFFile(pair.quantized) as quantized:
            v1 = create_plan(reference, quantized, [Q_NAME], provenance(), 4096)
        with self.models() as (_, reference, quantized):
            v2 = self.plan(reference, quantized)
            self.reject("repair_invalid_plan", validate_plan, v1, profile="v2")
            self.reject("repair_invalid_plan", validate_plan, v2)
            self.reject("repair_invalid_plan", write_candidate, reference, quantized, v1,
                        self.root / "cross-version", reserve_bytes=0, profile="v2")
            self.assertFalse((self.root / "cross-version").exists())

    def test_reference_inventory_cannot_contain_opaque_q8(self):
        tensors = reference_tensors()
        tensors[-1] = replace(tensors[-1], kind=Q8_0, payload=OPAQUE_Q8)
        metadata = replace_metadata(reference_metadata(), "general.quantization_version", 4, 2)
        with self.models(reference=tensors, reference_meta=metadata) as (_, reference, quantized):
            self.reject("repair_reference_unsupported", pair_inputs, reference, quantized, profile="v2")

    def test_v2_selection_refuses_q8_quantized_tensors_and_f16_references(self):
        tensors = quantized_tensors()
        tensors[0] = replace(tensors[0], kind=Q8_0, payload=OPAQUE_Q8)
        with self.models(quantized=tensors) as (_, reference, quantized):
            self.reject("repair_selection_invalid", self.plan, reference, quantized)
        ref = reference_tensors()
        ref[0] = replace(ref[0], kind=F16, payload=struct.pack("<64e", *([1.0] * 64)))
        with self.models(reference=ref) as (_, reference, quantized):
            self.reject("repair_selection_invalid", self.plan, reference, quantized)

    def test_pairing_checks_encoded_metadata_by_key_without_requiring_the_same_order(self):
        with self.models(reference_meta=list(reversed(reference_metadata()))) as (_, reference, quantized):
            self.assertEqual(pair_inputs(reference, quantized, profile="v2")["tensor_count"], 5)
        variants = [
            replace_metadata(quantized_metadata(), "general.name", 8, "another authored fixture"),
            [entry for entry in quantized_metadata() if entry[0] != "tokenizer.ggml.bos_token_id"],
        ]
        for metadata in variants:
            with self.subTest(metadata=metadata):
                with self.models(quantized_meta=metadata) as (_, reference, quantized):
                    self.reject("repair_pair_mismatch", pair_inputs, reference, quantized, profile="v2")

    def test_pairing_rejects_changed_complete_tensor_names_or_dimensions(self):
        for row in (replace(quantized_tensors()[0], name="blk.1.attn_q.weight"),
                    replace(quantized_tensors()[0], dimensions=(64, 1))):
            tensors = quantized_tensors(); tensors[0] = row
            with self.subTest(row=row):
                with self.models(quantized=tensors) as (_, reference, quantized):
                    self.reject("repair_pair_mismatch", pair_inputs, reference, quantized, profile="v2")

    def test_every_unquantized_f32_and_f16_type_and_payload_binding_is_required(self):
        for name, mutate_type in ((NORM_NAME, False), (F16_NAME, False), (NORM_NAME, True)):
            tensors = quantized_tensors()
            index = next(i for i, row in enumerate(tensors) if row.name == name)
            row = tensors[index]
            if mutate_type:
                tensors[index] = replace(row, kind=F16, payload=struct.pack("<2e", 1, -1))
            else:
                changed = bytearray(row.payload); changed[-1] ^= 1
                tensors[index] = replace(row, payload=bytes(changed))
            with self.subTest(name=name, mutate_type=mutate_type):
                with self.models(quantized=tensors) as (_, reference, quantized):
                    self.reject("repair_pair_mismatch", pair_inputs, reference, quantized, profile="v2")

    def test_dense_layout_fences_remain_required_under_v2(self):
        tensors = reference_tensors(); quant = quantized_tensors()
        tensors[-1] = replace(tensors[-1], name="vision.weight")
        quant[-1] = replace(quant[-1], name="vision.weight")
        with self.models(reference=tensors, quantized=quant) as (_, reference, quantized):
            self.reject("repair_layout_unsupported", self.plan, reference, quantized)
        for key, kind, value in (("llama.expert_count", 4, 1), ("general.architecture", 8, "gpt2")):
            with self.models(reference_meta=replace_metadata(reference_metadata(), key, kind, value),
                             quantized_meta=replace_metadata(quantized_metadata(), key, kind, value)) as (_, reference, quantized):
                code = "repair_architecture_unsupported" if key == "general.architecture" else "repair_layout_unsupported"
                self.reject(code, self.plan, reference, quantized)

    def test_work_and_growth_budgets_are_checked_before_selected_value_decoding(self):
        with self.models() as (_, reference, quantized):
            with patch.object(GGUFFile, "iter_values", side_effect=AssertionError("over-budget decode")) as decode:
                self.reject("repair_budget_exceeded", self.plan, reference, quantized, elements=63)
                self.reject("repair_budget_exceeded", self.plan, reference, quantized, added=219)
                self.reject("repair_budget_exceeded", self.plan, reference, quantized,
                            names=[Q_NAME, UP_NAME], elements=64)
            decode.assert_not_called()

    def test_selected_f32_values_and_q4_scales_must_be_finite(self):
        for kind, prefix in ((F32, struct.pack("<f", float("nan"))),
                             (F32, struct.pack("<f", float("inf"))),
                             (Q4_0, bytes.fromhex("00 7e")), (Q4_0, bytes.fromhex("00 7c"))):
            ref, quant = reference_tensors(), quantized_tensors()
            tensors = ref if kind == F32 else quant
            tensors[0] = replace(tensors[0], payload=prefix + tensors[0].payload[len(prefix):])
            with self.subTest(kind=kind, prefix=prefix):
                with self.models(reference=ref, quantized=quant) as (_, reference, quantized):
                    self.reject("gguf_nonfinite_weight", self.plan, reference, quantized)

    def test_forged_source_and_selected_hashes_cannot_write_a_candidate(self):
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            for source, field in (("reference", "sha256"), ("quantized", "sha256"),
                                  ("selected", "reference_sha256"), ("selected", "quantized_sha256")):
                forged = deepcopy(plan)
                (forged["selected"][0] if source == "selected" else forged[source])[field] = "0" * 64
                self.assertEqual(validate_plan(forged, profile="v2"), forged)
                output = self.root / ("forged-" + source + "-" + field)
                self.reject("repair_plan_mismatch", write_candidate, reference, quantized, forged,
                            output, reserve_bytes=0, profile="v2")
                self.assertFalse(output.exists())

    def test_changed_source_before_or_during_v2_copy_never_publishes_completion(self):
        output = self.root / "changed-before"
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized)
            changed = bytearray(pair.quantized_bytes); changed[-1] ^= 1
            pair.quantized.write_bytes(changed)
            self.reject("gguf_source_changed", write_candidate, reference, quantized, plan,
                        output, reserve_bytes=0, profile="v2")
        self.assertFalse(output.exists())
        output = self.root / "changed-during"
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized); actual = GGUFFile.iter_tensor_bytes; mutated = []
            def change_during_copy(model, tensor, chunk_bytes=None):
                for chunk in actual(model, tensor, chunk_bytes=chunk_bytes):
                    yield chunk
                    if model is reference and (output / "candidate.gguf.partial").exists() and not mutated:
                        changed = bytearray(pair.reference_bytes); changed[-1] ^= 1
                        old = pair.reference.stat(); pair.reference.write_bytes(changed)
                        os.utime(pair.reference, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
                        mutated.append(True)
            with patch.object(GGUFFile, "iter_tensor_bytes", new=change_during_copy):
                with self.assertRaises(QuantopsyError):
                    write_candidate(reference, quantized, plan, output, reserve_bytes=0, profile="v2")
            self.assertEqual(mutated, [True])
        self.assertFalse(output.exists())

    def test_v2_writer_reparses_both_stages_and_preserves_every_prescribed_byte(self):
        output = self.root / "complete"; events = []; readers = []
        real_link, actual_reader = os.link, GGUFFile
        def observe_reader(path, *args, **kwargs):
            if Path(path).name.startswith("candidate.gguf"):
                readers.append((Path(path).name, kwargs.get("profile")))
            return actual_reader(path, *args, **kwargs)
        def observe_install(source, destination, *args, **kwargs):
            self.assertFalse((output / "repair.json").exists())
            events.append((source, destination))
            if destination == "repair.json":
                with actual_reader(output / "candidate.gguf", profile="v2") as verified:
                    self.assertEqual(verified.tensor(Q_NAME).kind, F32)
                    self.assertEqual(verified.hash_tensor(verified.tensor(OPAQUE_NAME)), sha256(OPAQUE_Q8))
            return real_link(source, destination, *args, **kwargs)
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized)
            with patch("quantopsy.repair.GGUFFile", side_effect=observe_reader), \
                    patch("quantopsy.repair.os.link", side_effect=observe_install):
                manifest = write_candidate(reference, quantized, plan, output, reserve_bytes=0, profile="v2")
            self.assertEqual(pair.reference.read_bytes(), pair.reference_bytes)
            self.assertEqual(pair.quantized.read_bytes(), pair.quantized_bytes)
            with actual_reader(output / "candidate.gguf", profile="v2") as candidate:
                for tensor in candidate.tensors:
                    source = reference if tensor.name == Q_NAME else quantized
                    self.assertEqual(candidate.hash_tensor(tensor), source.hash_tensor(source.tensor(tensor.name)))
                keys = [key for key in quantized.metadata if key != "general.file_type"]
                self.assertEqual(list(candidate.metadata), keys)
                self.assertTrue(all(candidate.metadata[k].encoded == quantized.metadata[k].encoded for k in keys))
                self.assertEqual(candidate.metadata["general.quantization_version"].value, 2)
        self.assertEqual(readers, [("candidate.gguf.partial", "v2"), ("candidate.gguf", "v2")])
        self.assertEqual(events, [("candidate.gguf.partial", "candidate.gguf"), ("repair.json.partial", "repair.json")])
        self.assertEqual((output / "candidate.gguf").read_bytes(), expected_candidate_bytes())
        self.assertEqual((output / "plan.json").read_bytes(), canonical_json_bytes(plan))
        self.assertEqual(json.loads((output / "repair.json").read_text()), manifest)
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["plan_sha256"], sha256(canonical_json_bytes(plan)))
        self.assertEqual(manifest["output"]["sha256"], sha256(expected_candidate_bytes()))
        self.assertEqual(manifest["metadata_omissions"], ["general.file_type"])
        self.assertEqual(manifest["behavioral_validation"], "not_run")
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        for path in output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_v2_create_only_outputs_preserve_existing_files_directories_and_symlinks(self):
        directory = self.root / "prior"; directory.mkdir()
        sentinel = directory / "keep"; sentinel.write_bytes(b"prior authored bytes")
        empty = self.root / "empty"; empty.mkdir()
        regular = self.root / "regular"; regular.write_bytes(b"prior file")
        link = self.root / "link"; link.symlink_to(directory, target_is_directory=True)
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            for destination in (directory, empty, regular, link):
                self.reject("repair_output_exists", write_candidate, reference, quantized, plan,
                            destination, reserve_bytes=0, profile="v2")
        self.assertEqual(sentinel.read_bytes(), b"prior authored bytes")
        self.assertEqual(regular.read_bytes(), b"prior file")
        self.assertEqual(list(empty.iterdir()), [])
        self.assertTrue(link.is_symlink())

    def test_v2_output_and_storage_budgets_fail_before_directory_creation(self):
        output = self.root / "budget"
        with self.models(limits=Limits(max_output_bytes=1)) as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            self.reject("repair_output_limit", write_candidate, reference, quantized, plan,
                        output, reserve_bytes=0, profile="v2")
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            with patch("quantopsy.repair.os.fstatvfs", return_value=SimpleNamespace(f_bavail=0, f_frsize=4096)):
                self.reject("repair_storage_insufficient", write_candidate, reference, quantized, plan,
                            output, reserve_bytes=0, profile="v2")
        self.assertFalse(output.exists())

    def test_manifest_install_failure_removes_owned_v2_artifacts_without_completion(self):
        output = self.root / "manifest-failure"; real_link = os.link; attempted = []
        def fail_manifest(source, destination, *args, **kwargs):
            if destination == "repair.json":
                attempted.append(True)
                self.assertTrue((output / "candidate.gguf").is_file())
                self.assertFalse((output / "repair.json").exists())
                raise OSError("authored manifest failure")
            return real_link(source, destination, *args, **kwargs)
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            with patch("quantopsy.repair.os.link", side_effect=fail_manifest):
                self.reject("repair_output_failed", write_candidate, reference, quantized, plan,
                            output, reserve_bytes=0, profile="v2")
        self.assertEqual(attempted, [True])
        self.assertFalse(output.exists())

    def test_failed_v2_install_preserves_replaced_inodes_and_unexpected_files(self):
        output = self.root / "foreign"; replacement = self.root / "replacement"
        replacement.write_bytes(b"independent replacement inode")
        def replace_then_fail(source, destination, *args, **kwargs):
            self.assertEqual(destination, "candidate.gguf")
            (output / "keep.txt").write_bytes(b"unrelated authored file")
            os.replace(replacement, output / "candidate.gguf.partial")
            raise OSError("authored install failure")
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            with patch("quantopsy.repair.os.link", side_effect=replace_then_fail):
                self.reject("repair_output_failed", write_candidate, reference, quantized, plan,
                            output, reserve_bytes=0, profile="v2")
        self.assertEqual(set(p.name for p in output.iterdir()), {"candidate.gguf.partial", "keep.txt"})
        self.assertEqual((output / "candidate.gguf.partial").read_bytes(), b"independent replacement inode")
        self.assertFalse((output / "repair.json").exists())

    def test_cli_v2_inspection_is_opt_in_and_diagnostics_reject_profile_flags(self):
        pair = write_v2_pair(self.folder())
        status, stdout, stderr = self.command("inspect", pair.quantized)
        self.assertEqual((status, stdout), (2, ""))
        self.assertEqual(json.loads(stderr)["error"]["code"], "gguf_tensor_type_unsupported")
        status, stdout, stderr = self.command("inspect", pair.quantized, "--profile", "v2")
        self.assertEqual((status, stderr), (0, ""))
        report = json.loads(stdout)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["tensor_types"], {"Q4_0": 2, "F32": 1, "F16": 1, "Q8_0": 1})
        self.assertEqual(report["file"]["sha256"], sha256(pair.quantized_bytes))
        for command, option in (("diagnose", "--tensor"), ("probe", "--probes")):
            status, stdout, stderr = self.command(command, pair.reference, pair.quantized,
                                                  option, Q_NAME, "--profile", "v2")
            self.assertEqual((status, stdout), (2, ""))
            self.assertEqual(json.loads(stderr)["error"]["code"], "invalid_arguments")

    def test_cli_plan_and_repair_require_explicit_matching_version_and_publish_schema_two(self):
        pair = write_v2_pair(self.folder())
        declared = self.root / "provenance.json"; declared.write_bytes(canonical_json_bytes(provenance()))
        plan_path, output = self.root / "v2-plan.json", self.root / "cli-complete"
        status, stdout, stderr = self.command("plan", pair.reference, pair.quantized,
            "--profile", "v2", "--tensor", Q_NAME, "--provenance", declared,
            "--max-added-bytes", 4096, "--output", plan_path)
        self.assertEqual((status, stderr), (0, ""))
        plan = json.loads(plan_path.read_text())
        self.assertEqual(plan["schema_version"], 2)
        status, stdout, stderr = self.command("repair", pair.reference, pair.quantized,
            "--plan", plan_path, "--output-dir", self.root / "implicit", "--reserve-bytes", 0)
        self.assertEqual((status, stdout), (2, ""))
        self.assertFalse((self.root / "implicit").exists())
        status, stdout, stderr = self.command("repair", pair.reference, pair.quantized,
            "--profile", "v2", "--plan", plan_path, "--output-dir", output, "--reserve-bytes", 0)
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(json.loads(stdout), json.loads((output / "repair.json").read_text()))
        self.assertEqual(json.loads(stdout)["schema_version"], 2)
        self.assertEqual((output / "candidate.gguf").read_bytes(), expected_candidate_bytes())
        forged = deepcopy(plan); forged["schema_version"] = 1
        wrong = self.root / "wrong-plan.json"; wrong.write_bytes(canonical_json_bytes(forged))
        status, stdout, stderr = self.command("repair", pair.reference, pair.quantized,
            "--profile", "v2", "--plan", wrong, "--output-dir", self.root / "wrong-schema", "--reserve-bytes", 0)
        self.assertEqual((status, stdout), (2, ""))
        self.assertEqual(json.loads(stderr)["error"]["code"], "repair_invalid_plan")
        self.assertFalse((self.root / "wrong-schema").exists())
