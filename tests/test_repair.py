"""Twenty-two authored compatibility and real tiny-container rebuild cases."""

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from quantopsy.contracts import Limits
from quantopsy.errors import QuantopsyError
from quantopsy.gguf import GGUFFile
from quantopsy.repair import create_plan, pair_inputs, validate_plan, write_candidate
from tests.fixtures import (
    F16, F32, NORM_NAME, Q4_0, Q4_POSITIVE, Q_NAME, UP_NAME, Tensor,
    identity_metadata, provenance, quantized_tensors, reference_tensors,
    replace_metadata, sha256, write_pair,
)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="quantopsy-authored-repair-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @contextmanager
    def models(self, *, limits=Limits(), **kwargs):
        pair = write_pair(self.root, **kwargs)
        with GGUFFile(pair.reference, limits=limits) as reference:
            with GGUFFile(pair.quantized, limits=limits) as quantized:
                yield pair, reference, quantized

    def plan(self, reference, quantized, names=None, *, added=1024, elements=96):
        return create_plan(reference, quantized, [Q_NAME] if names is None else names,
                           provenance(), added, max_elements=elements)

    def test_pair_summary_binds_complete_original_bytes_and_honest_lineage(self):
        with self.models() as (pair, reference, quantized):
            summary = pair_inputs(reference, quantized)
            self.assertEqual(summary, {
                "architecture": "llama", "tensor_count": 3,
                "reference": {"name": "reference.gguf", "sha256": sha256(pair.reference_bytes), "size": len(pair.reference_bytes)},
                "quantized": {"name": "quantized.gguf", "sha256": sha256(pair.quantized_bytes), "size": len(pair.quantized_bytes)},
                "lineage": "compatibility_checked_lineage_unverified",
            })
            self.assertNotIn(str(self.root), json.dumps(summary))
            # Pairing does not take ownership of either descriptor.
            self.assertEqual(reference.sha256(), sha256(pair.reference_bytes))
            self.assertEqual(quantized.sha256(), sha256(pair.quantized_bytes))

    def test_pair_rejects_architecture_and_nonquantization_identity_disagreement(self):
        changes = [("general.architecture", 8, "gpt2"),
                   ("general.name", 8, "A different model"),
                   ("llama.context_length", 4, 128),
                   ("tokenizer.ggml.tokens", 9, (8, ["authored-a", "different"]))]
        for key, kind, value in changes:
            with self.subTest(key=key):
                changed = replace_metadata(identity_metadata(quantized=True), key, kind, value)
                with self.models(quantized_meta=changed) as (_, reference, quantized):
                    with self.assertRaises(QuantopsyError) as caught: pair_inputs(reference, quantized)
                    self.assertEqual(caught.exception.code, "repair_pair_mismatch")
        for metadata in [identity_metadata(quantized=True) + [("fixture.extra", 8, "not shared")],
                         [entry for entry in identity_metadata(quantized=True) if entry[0] != "general.basename"]]:
            with self.models(quantized_meta=metadata) as (_, reference, quantized):
                with self.assertRaises(QuantopsyError): pair_inputs(reference, quantized)

    def test_quantization_only_metadata_may_differ_without_requiring_table_order(self):
        ref_meta = identity_metadata() + [("general.quantized_by", 8, "reference author")]
        quant_meta = identity_metadata(quantized=True) + [
            ("general.quantized_by", 8, "quantized author"),
            ("general.quantized_at", 8, "authored-not-wall-clock"),
        ]
        with self.models(reference_meta=list(reversed(ref_meta)), quantized_meta=quant_meta) as (_, reference, quantized):
            self.assertEqual(pair_inputs(reference, quantized)["tensor_count"], 3)
            self.assertEqual(self.plan(reference, quantized)["behavioral_validation"], "not_run")

    def test_pair_requires_complete_tensor_names_and_exact_dimensions(self):
        ordinary = quantized_tensors()
        variants = [ordinary[:-1], ordinary + [Tensor("extra.weight", (1,), F32, struct.pack("<f", 0))],
                    [Tensor("blk.1.attn_q.weight", ordinary[0].dimensions, Q4_0, ordinary[0].payload), *ordinary[1:]],
                    [Tensor(Q_NAME, (64, 1), Q4_0, ordinary[0].payload), *ordinary[1:]]]
        for tensors in variants:
            with self.subTest(names=[tensor.name for tensor in tensors]):
                with self.models(quantized=tensors) as (_, reference, quantized):
                    with self.assertRaises(QuantopsyError) as caught: pair_inputs(reference, quantized)
                    self.assertEqual(caught.exception.code, "repair_pair_mismatch")

    def test_unquantized_tensors_must_preserve_encoding_and_payload_exactly(self):
        variants = [Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 1, -2)),
                    Tensor(NORM_NAME, (2,), F16, struct.pack("<2e", 1, -1))]
        for norm in variants:
            with self.models(quantized=[*quantized_tensors()[:2], norm]) as (_, reference, quantized):
                with self.assertRaises(QuantopsyError) as caught: pair_inputs(reference, quantized)
                self.assertEqual(caught.exception.code, "repair_pair_mismatch")

    def test_reference_quantization_and_changed_input_identity_are_refused(self):
        with self.models(reference=quantized_tensors(), reference_meta=identity_metadata(quantized=True)) as (_, reference, quantized):
            with self.assertRaises(QuantopsyError) as caught: pair_inputs(reference, quantized)
            self.assertEqual(caught.exception.code, "repair_reference_unsupported")
        with self.models() as (pair, reference, quantized):
            original = pair.reference.stat()
            data = bytearray(pair.reference_bytes); data[-1] ^= 1
            pair.reference.write_bytes(data)
            os.utime(pair.reference, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
            with self.assertRaises(QuantopsyError) as caught: pair_inputs(reference, quantized)
            self.assertEqual(caught.exception.code, "gguf_source_changed")

    def test_dense_llama_profile_rejects_other_architectures_experts_and_fused_layouts(self):
        ref_meta = replace_metadata(identity_metadata(), "general.architecture", 8, "gpt2")
        quant_meta = replace_metadata(identity_metadata(quantized=True), "general.architecture", 8, "gpt2")
        with self.models(reference_meta=ref_meta, quantized_meta=quant_meta) as (_, reference, quantized):
            self.assertEqual(pair_inputs(reference, quantized)["architecture"], "gpt2")
            with self.assertRaises(QuantopsyError) as caught: self.plan(reference, quantized)
            self.assertEqual(caught.exception.code, "repair_architecture_unsupported")
            with self.assertRaises(QuantopsyError): pair_inputs(reference, quantized, require_llama="yes")
        for key in ["llama.expert_count", "llama.expert_used_count", "llama.moe_layout"]:
            with self.models(reference_meta=identity_metadata() + [(key, 4, 1)],
                             quantized_meta=identity_metadata(quantized=True) + [(key, 4, 1)]) as (_, reference, quantized):
                with self.assertRaises(QuantopsyError) as caught: self.plan(reference, quantized)
                self.assertEqual(caught.exception.code, "repair_layout_unsupported")
        for name, shape in [("blk.0.attn_qkv.weight", (1, 1)), ("blk.0.lora.weight", (1, 1)),
                            ("vision.weight", (1, 1)), ("fixture.rank3", (1, 1, 1))]:
            extra = Tensor(name, shape, F32, struct.pack("<f", 1))
            with self.models(reference=reference_tensors() + [extra], quantized=quantized_tensors() + [extra]) as (_, reference, quantized):
                with self.assertRaises(QuantopsyError) as caught: self.plan(reference, quantized)
                self.assertEqual(caught.exception.code, "repair_layout_unsupported")
        with self.models(reference_meta=identity_metadata() + [("llama.expert_count", 4, 0)],
                         quantized_meta=identity_metadata(quantized=True) + [("llama.expert_count", 4, 0)]) as (_, reference, quantized):
            self.assertEqual(len(self.plan(reference, quantized)["selected"]), 1)

    def test_explicit_plan_binds_selected_bytes_shapes_input_hashes_and_provenance(self):
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized, added=92, elements=64)
            self.assertEqual(plan["kind"], "quantopsy.repair-plan")
            self.assertEqual(plan["schema_version"], 1)
            self.assertEqual(plan["architecture"], "llama")
            self.assertEqual(plan["selected"], [{
                "name": Q_NAME, "dimensions": [32, 2], "elements": 64,
                "reference_kind": 1, "quantized_kind": 2,
                "reference_sha256": sha256(reference_tensors()[0].payload),
                "quantized_sha256": sha256(quantized_tensors()[0].payload),
                "reference_bytes": 128, "quantized_bytes": 36,
            }])
            self.assertEqual((plan["added_weight_bytes"], plan["max_added_bytes"], plan["max_elements"]), (92, 92, 64))
            self.assertEqual(plan["reference"]["sha256"], sha256(pair.reference_bytes))
            self.assertEqual(plan["quantized"]["sha256"], sha256(pair.quantized_bytes))
            self.assertEqual(plan["provenance"], provenance())
            self.assertEqual(plan["lineage"], "operator_declared_compatibility_checked")
            self.assertEqual(plan["behavioral_validation"], "not_run")

    def test_selected_matrix_encoding_rank_and_both_inputs_finiteness_are_checked(self):
        variants = [
            ([Tensor(Q_NAME, (32, 2), F32, struct.pack("<64f", *([1] * 64))), *reference_tensors()[1:]], quantized_tensors()),
            ([Tensor(Q_NAME, (32,), F16, struct.pack("<32e", *([1] * 32))), *reference_tensors()[1:]],
             [Tensor(Q_NAME, (32,), Q4_0, Q4_POSITIVE), *quantized_tensors()[1:]]),
            ([Tensor(Q_NAME, (32, 2), F16, struct.pack("<e", float("nan")) + reference_tensors()[0].payload[2:]), *reference_tensors()[1:]], quantized_tensors()),
            (reference_tensors(), [Tensor(Q_NAME, (32, 2), Q4_0, bytes.fromhex("00 7c") + quantized_tensors()[0].payload[2:]), *quantized_tensors()[1:]]),
        ]
        for original, encoded in variants:
            with self.subTest(reference_kind=original[0].kind, shape=original[0].dimensions, scale=encoded[0].payload[:2]):
                with self.models(reference=original, quantized=encoded) as (_, reference, quantized):
                    with self.assertRaises(QuantopsyError): self.plan(reference, quantized)

    def test_selection_requires_explicit_unique_present_supported_matrix_names(self):
        with self.models() as (_, reference, quantized):
            for names in [[], [Q_NAME, Q_NAME], [NORM_NAME], ["blk.9.attn_q.weight"], (Q_NAME,), [True]]:
                with self.subTest(names=names):
                    with self.assertRaises(QuantopsyError): self.plan(reference, quantized, names)

    def test_provenance_requires_exact_nonblank_bounded_utf8_fields(self):
        variants = [{key: value for key, value in provenance().items() if key != "license"},
                    {**provenance(), "unrequested": "extra"}]
        variants += [{**provenance(), "revision": value} for value in ["", " ", True, "bad\0revision", "\ud800", "é" * 1025]]
        with self.models() as (_, reference, quantized):
            for declared in variants:
                with self.subTest(keys=list(declared), revision=repr(declared.get("revision"))[:30]):
                    with self.assertRaises(QuantopsyError): create_plan(reference, quantized, [Q_NAME], declared, 1024)
            declared = provenance(); declared["revision"] = "é" * 1024
            self.assertEqual(create_plan(reference, quantized, [Q_NAME], declared, 1024)["provenance"]["revision"], declared["revision"])

    def test_added_weight_and_decode_work_budgets_accept_exact_bounds_before_decode(self):
        with self.models() as (_, reference, quantized):
            self.assertEqual(self.plan(reference, quantized, added=92, elements=64)["added_weight_bytes"], 92)
            with patch.object(GGUFFile, "iter_values", side_effect=AssertionError("over-budget work must not decode")):
                for added, elements in [(91, 64), (92, 63), (1024, 95)]:
                    names = [Q_NAME, UP_NAME] if elements == 95 else [Q_NAME]
                    with self.assertRaises(QuantopsyError) as caught: self.plan(reference, quantized, names, added=added, elements=elements)
                    self.assertEqual(caught.exception.code, "repair_budget_exceeded")
            for added, elements in [(-1, 64), (True, 64), (1.5, 64), (92, 0), (92, True), (92, 64_000_001)]:
                with self.assertRaises(QuantopsyError): self.plan(reference, quantized, added=added, elements=elements)
        with self.models(limits=Limits(max_selected_elements=63)) as (_, reference, quantized):
            with self.assertRaises(QuantopsyError): self.plan(reference, quantized, elements=64)

    def test_plan_validation_is_detached_strict_and_checks_internal_arithmetic(self):
        with self.models() as (_, reference, quantized): plan = self.plan(reference, quantized)
        accepted = validate_plan(plan)
        plan["selected"][0]["dimensions"][0] = 64
        plan["provenance"]["license"] = "changed after validation"
        self.assertEqual(accepted["selected"][0]["dimensions"], [32, 2])
        self.assertEqual(accepted["provenance"]["license"], "authored fixture only")
        variants = []
        for record_path in [(), ("selected", 0), ("reference",), ("provenance",)]:
            changed = deepcopy(accepted); record = changed
            for key in record_path: record = record[key]
            record["extra"] = "not supported"; variants.append(changed)
        for field, value in [("elements", 32), ("dimensions", [32, 1]), ("reference_bytes", 126),
                             ("quantized_bytes", 18), ("reference_kind", True), ("reference_sha256", "a" * 63)]:
            changed = deepcopy(accepted); changed["selected"][0][field] = value; variants.append(changed)
        for field, value in [("schema_version", True), ("added_weight_bytes", 91),
                             ("lineage", "verified"), ("behavioral_validation", "passed")]:
            changed = deepcopy(accepted); changed[field] = value; variants.append(changed)
        changed = deepcopy(accepted); changed["reference"]["name"] = "../reference.gguf"; variants.append(changed)
        for value in variants:
            with self.assertRaises(QuantopsyError): validate_plan(value)

    def test_semantically_valid_forged_source_or_tensor_binding_cannot_write(self):
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            for label in ["reference", "quantized", "reference_sha256", "quantized_sha256"]:
                changed = deepcopy(plan)
                if label in ("reference", "quantized"): changed[label]["sha256"] = "0" * 64
                else: changed["selected"][0][label] = "0" * 64
                self.assertEqual(validate_plan(changed), changed)
                output = self.root / ("forged-" + label)
                with self.assertRaises(QuantopsyError) as caught: write_candidate(reference, quantized, changed, output, reserve_bytes=0)
                self.assertEqual(caught.exception.code, "repair_plan_mismatch")
                self.assertFalse(output.exists())

    def test_candidate_restores_only_selected_reference_bytes_and_preserves_metadata(self):
        output = self.root / "single-restoration"
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized)
            manifest = write_candidate(reference, quantized, plan, output, reserve_bytes=0)
            with GGUFFile(output / "candidate.gguf") as candidate:
                self.assertEqual([item.name for item in candidate.tensors], [Q_NAME, UP_NAME, NORM_NAME])
                self.assertEqual([item.kind for item in candidate.tensors], [F16, Q4_0, F32])
                self.assertEqual(b"".join(candidate.iter_tensor_bytes(candidate.tensor(Q_NAME))), reference_tensors()[0].payload)
                for expected in quantized_tensors()[1:]:
                    self.assertEqual(b"".join(candidate.iter_tensor_bytes(candidate.tensor(expected.name))), expected.payload)
                keys = [key for key in quantized.metadata if key != "general.file_type"]
                self.assertEqual(list(candidate.metadata), keys)
                for key in keys: self.assertEqual(candidate.metadata[key].encoded, quantized.metadata[key].encoded)
                self.assertEqual(candidate.metadata["general.quantization_version"].value, 2)
                self.assertEqual(candidate.sha256(), manifest["output"]["sha256"])
            self.assertEqual(pair.reference.read_bytes(), pair.reference_bytes)
            self.assertEqual(pair.quantized.read_bytes(), pair.quantized_bytes)
        self.assertEqual(set(path.name for path in output.iterdir()), {"plan.json", "candidate.gguf", "repair.json"})

    def test_all_selected_matrices_use_reference_encoding_without_reordering_inventory(self):
        output = self.root / "both-restored"
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized, [UP_NAME, Q_NAME], added=138, elements=96)
            self.assertEqual([row["name"] for row in plan["selected"]], [UP_NAME, Q_NAME])
            write_candidate(reference, quantized, plan, output, reserve_bytes=0)
        with GGUFFile(output / "candidate.gguf") as candidate:
            self.assertEqual([item.name for item in candidate.tensors], [Q_NAME, UP_NAME, NORM_NAME])
            self.assertEqual([item.kind for item in candidate.tensors], [F16, F16, F32])
            for expected in reference_tensors():
                self.assertEqual(b"".join(candidate.iter_tensor_bytes(candidate.tensor(expected.name))), expected.payload)
                self.assertEqual(candidate.hash_tensor(candidate.tensor(expected.name)), sha256(expected.payload))

    def test_non_power_of_two_alignment_is_preserved_with_rebuilt_offsets(self):
        output = self.root / "aligned-24"
        with self.models(alignment=24) as (_, reference, quantized):
            write_candidate(reference, quantized, self.plan(reference, quantized), output, reserve_bytes=0)
        with GGUFFile(output / "candidate.gguf") as candidate:
            self.assertEqual(candidate.alignment, 24)
            self.assertEqual(candidate.data_offset % 24, 0)
            self.assertEqual([tensor.offset for tensor in candidate.tensors], [0, 144, 168])
            self.assertEqual(tuple(candidate.iter_values(candidate.tensor(Q_NAME))), tuple(index / 4 for index in range(64)))
            self.assertEqual(candidate.file_size, candidate.data_offset + 176)

    def test_manifest_is_published_last_after_reparse_and_contains_exact_artifact_hashes(self):
        output = self.root / "manifest-order"; events = []; real_link = os.link
        def observe_install(source, destination, *args, **kwargs):
            events.append((str(source), str(destination)))
            self.assertFalse((output / "repair.json").exists())
            if destination == "repair.json":
                self.assertTrue((output / "candidate.gguf").is_file())
                with GGUFFile(output / "candidate.gguf") as candidate:
                    self.assertEqual(candidate.tensor(Q_NAME).kind, F16)
            return real_link(source, destination, *args, **kwargs)
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            with patch("quantopsy.repair.os.link", side_effect=observe_install):
                manifest = write_candidate(reference, quantized, plan, output, reserve_bytes=0)
        self.assertEqual(events, [("candidate.gguf.partial", "candidate.gguf"), ("repair.json.partial", "repair.json")])
        self.assertEqual(json.loads((output / "repair.json").read_text(encoding="utf-8")), manifest)
        expected_plan = (json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        self.assertEqual((output / "plan.json").read_bytes(), expected_plan)
        self.assertEqual(manifest["plan_sha256"], sha256(expected_plan))
        candidate_bytes = (output / "candidate.gguf").read_bytes()
        self.assertEqual(manifest["output"], {"name": "candidate.gguf", "sha256": sha256(candidate_bytes), "size": len(candidate_bytes)})
        self.assertEqual(manifest["selected"], plan["selected"])
        self.assertEqual(manifest["metadata_omissions"], ["general.file_type"])
        self.assertEqual(manifest["provenance"], provenance())
        self.assertEqual(manifest["behavioral_validation"], "not_run")
        self.assertEqual(manifest["verification"], {"container_reparsed": True, "metadata_preserved": True,
                                                   "tensor_payloads_verified": True, "source_hashes_verified": True})
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        for path in output.iterdir(): self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_existing_directory_file_and_symlink_outputs_are_never_overwritten(self):
        directory = self.root / "existing-directory"; directory.mkdir(); sentinel = directory / "keep"; sentinel.write_bytes(b"keep")
        regular = self.root / "existing-file"; regular.write_bytes(b"existing file")
        linked = self.root / "existing-link"; linked.symlink_to(directory, target_is_directory=True)
        empty = self.root / "existing-empty"; empty.mkdir()
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            for destination in [directory, regular, linked, empty]:
                with self.assertRaises(QuantopsyError) as caught: write_candidate(reference, quantized, plan, destination, reserve_bytes=0)
                self.assertEqual(caught.exception.code, "repair_output_exists")
        self.assertEqual(sentinel.read_bytes(), b"keep")
        self.assertEqual(regular.read_bytes(), b"existing file")
        self.assertTrue(linked.is_symlink())
        self.assertEqual(list(empty.iterdir()), [])
        self.assertEqual(set(path.name for path in directory.iterdir()), {"keep"})

    def test_output_size_storage_and_reserve_budgets_fail_before_bundle_creation(self):
        output = self.root / "budget-output"
        with self.models(limits=Limits(max_output_bytes=1)) as (_, reference, quantized):
            with self.assertRaises(QuantopsyError) as caught:
                write_candidate(reference, quantized, self.plan(reference, quantized), output, reserve_bytes=0)
            self.assertEqual(caught.exception.code, "repair_output_limit")
        self.assertFalse(output.exists())
        with self.models() as (_, reference, quantized):
            plan = self.plan(reference, quantized)
            with patch("quantopsy.repair.os.fstatvfs", return_value=SimpleNamespace(f_bavail=0, f_frsize=4096)):
                with self.assertRaises(QuantopsyError) as caught: write_candidate(reference, quantized, plan, output, reserve_bytes=0)
                self.assertEqual(caught.exception.code, "repair_storage_insufficient")
            for reserve in [-1, True, 1.5, 2**53]:
                with self.assertRaises(QuantopsyError): write_candidate(reference, quantized, plan, output, reserve_bytes=reserve)
        self.assertFalse(output.exists())

    def test_input_mutation_before_or_during_copy_never_publishes_completion(self):
        output = self.root / "mutated-input"
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized)
            changed = bytearray(pair.quantized_bytes); changed[-1] ^= 1; pair.quantized.write_bytes(changed)
            with self.assertRaises(QuantopsyError): write_candidate(reference, quantized, plan, output, reserve_bytes=0)
        self.assertFalse(output.exists())
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized); actual_stream = GGUFFile.iter_tensor_bytes; mutated = []
            def change_during_copy(model, tensor, chunk_bytes=None):
                for chunk in actual_stream(model, tensor, chunk_bytes=chunk_bytes):
                    yield chunk
                    if model is reference and (output / "candidate.gguf.partial").exists() and not mutated:
                        mutated.append(True)
                        old = pair.reference.stat(); changed = bytearray(pair.reference_bytes); changed[-1] ^= 1
                        pair.reference.write_bytes(changed)
                        os.utime(pair.reference, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
            with patch.object(GGUFFile, "iter_tensor_bytes", new=change_during_copy):
                with self.assertRaises(QuantopsyError): write_candidate(reference, quantized, plan, output, reserve_bytes=0)
            self.assertEqual(mutated, [True])
        self.assertFalse(output.exists())

    def test_interruption_cleanup_removes_only_owned_inodes_and_keeps_unexpected_files(self):
        with self.models() as (pair, reference, quantized):
            plan = self.plan(reference, quantized)
            for stage in ["first-fsync", "manifest-link"]:
                output = self.root / stage
                real_link = os.link
                def interrupt_link(source, destination, *args, **kwargs):
                    if destination == "repair.json": raise KeyboardInterrupt("authored interruption")
                    return real_link(source, destination, *args, **kwargs)
                target = "quantopsy.repair.os.fsync" if stage == "first-fsync" else "quantopsy.repair.os.link"
                effect = KeyboardInterrupt("authored interruption") if stage == "first-fsync" else interrupt_link
                with patch(target, side_effect=effect):
                    with self.assertRaises(KeyboardInterrupt): write_candidate(reference, quantized, plan, output, reserve_bytes=0)
                self.assertFalse(output.exists())
            output = self.root / "foreign-content"
            replacement = self.root / "foreign-replacement"; replacement.write_bytes(b"unrelated replacement inode")
            def replace_before_install(source, destination, *args, **kwargs):
                self.assertEqual(destination, "candidate.gguf")
                (output / "keep.txt").write_bytes(b"unrelated new file")
                os.replace(replacement, output / "candidate.gguf.partial")
                raise OSError("authored install interruption")
            with patch("quantopsy.repair.os.link", side_effect=replace_before_install):
                with self.assertRaises(QuantopsyError) as caught: write_candidate(reference, quantized, plan, output, reserve_bytes=0)
                self.assertEqual(caught.exception.code, "repair_output_failed")
            self.assertEqual(set(path.name for path in output.iterdir()), {"candidate.gguf.partial", "keep.txt"})
            self.assertEqual((output / "candidate.gguf.partial").read_bytes(), b"unrelated replacement inode")
            self.assertEqual((output / "keep.txt").read_bytes(), b"unrelated new file")
            self.assertFalse((output / "repair.json").exists())
            self.assertEqual(pair.reference.read_bytes(), pair.reference_bytes)
            self.assertEqual(pair.quantized.read_bytes(), pair.quantized_bytes)
