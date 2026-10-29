"""Authored T2 numerical checks. UNRUN; no real model or inference runner."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from quantopsy.contracts import DEFAULT_LIMITS
from quantopsy.diagnostics import diagnose, probe
from quantopsy.errors import QuantopsyError
from quantopsy.gguf import GGUFFile
from tests.fixtures import (
    F16, F32, NORM_NAME, Q4_0, Q4_POSITIVE, Q_NAME, UP_NAME,
    Tensor, identity_metadata, quantized_tensors, reference_tensors,
    replace_metadata, sha256, write_pair,
)


def probe_document(*, tensor=Q_NAME, vectors=None):
    return {
        "schema_version": 1, "kind": "quantopsy.linear-probes", "tensor": tensor,
        "vectors": vectors if vectors is not None else [[1.0] + [0.0] * 31],
        "provenance": {"model_id": "authored/tiny-llama", "revision": "authored-r1",
                       "dataset": "authored calibration vectors", "capture": "independent authored coordinates"},
    }


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="quantopsy-diagnostics-")
        self.directory = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.sequence = 0

    def pair(self, **kwargs):
        self.sequence += 1
        directory = self.directory / str(self.sequence)
        directory.mkdir(mode=0o700)
        return write_pair(directory, **kwargs)

    @contextmanager
    def opened(self, pair, limits=DEFAULT_LIMITS):
        with GGUFFile(pair.reference, limits) as reference, GGUFFile(pair.quantized, limits) as quantized:
            yield reference, quantized

    def assert_code(self, code, operation):
        with self.assertRaises(QuantopsyError) as failure:
            operation()
        self.assertEqual(failure.exception.code, code)

    def test_constant_reference_against_zero_weights_has_independent_exact_metrics(self):
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            result = diagnose(reference, quantized, [UP_NAME])
        row = result["tensors"][0]
        self.assertEqual(row["name"], UP_NAME)
        self.assertEqual(row["dimensions"], [32, 1])
        self.assertEqual(row["elements"], 32)
        self.assertEqual(row["changed_values"], 32)
        self.assertEqual(row["rmse"], 1.25)
        self.assertEqual(row["reference_rms"], 1.25)
        self.assertEqual(row["max_absolute_error"], 1.25)
        self.assertEqual(row["relative_l2"], 1.0)
        self.assertIsNone(row["relative_l2_unavailable"])

    def test_signed_two_row_fixture_preserves_magnitude_without_position_or_sign_shortcuts(self):
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            result = diagnose(reference, quantized, [Q_NAME])
        row = result["tensors"][0]
        # Independently summed from the literal packed fixture: squared error
        # 6086 and reference energy 5334 across the two 32-value rows.
        self.assertEqual(row["changed_values"], 64)
        self.assertAlmostEqual(row["rmse"], math.sqrt(95.09375))
        self.assertAlmostEqual(row["reference_rms"], math.sqrt(83.34375))
        self.assertAlmostEqual(row["relative_l2"], math.sqrt(6086 / 5334))
        self.assertEqual(row["max_absolute_error"], 17.25)

    def test_selection_order_and_complete_input_hashes_do_not_claim_behavior_or_lineage(self):
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            result = diagnose(reference, quantized, [UP_NAME, Q_NAME])
        self.assertEqual([row["name"] for row in result["tensors"]], [UP_NAME, Q_NAME])
        self.assertEqual(result["selected_elements"], 96)
        self.assertEqual(result["inputs"]["reference"]["sha256"], sha256(pair.reference_bytes))
        self.assertEqual(result["inputs"]["quantized"]["sha256"], sha256(pair.quantized_bytes))
        self.assertEqual(result["inputs"]["lineage"], "compatibility_checked_lineage_unverified")
        self.assertEqual(result["evidence"], "parameter_error_only")
        self.assertEqual(result["behavioral_validation"], "not_run")
        self.assertNotIn(str(self.directory), json.dumps(result))
        self.assertEqual(pair.reference.read_bytes(), pair.reference_bytes)
        self.assertEqual(pair.quantized.read_bytes(), pair.quantized_bytes)

    def test_zero_reference_energy_stays_null_and_identical_zero_weights_stay_zero(self):
        zeros = [Tensor(UP_NAME, (32, 1), F16, struct.pack("<32e", *([0.0] * 32)))]
        nonzero = [Tensor(UP_NAME, (32, 1), Q4_0, Q4_POSITIVE)]
        pair = self.pair(reference=zeros, quantized=nonzero)
        with self.opened(pair) as (reference, quantized):
            row = diagnose(reference, quantized, [UP_NAME])["tensors"][0]
        self.assertGreater(row["rmse"], 0)
        self.assertIsNone(row["relative_l2"])
        self.assertEqual(row["relative_l2_unavailable"], "zero_reference_energy")
        same = self.pair(reference=zeros, quantized=zeros)
        with self.opened(same) as (reference, quantized):
            row = diagnose(reference, quantized, [UP_NAME])["tensors"][0]
        self.assertEqual(row["changed_values"], 0)
        self.assertEqual(row["rmse"], 0)
        self.assertIsNone(row["relative_l2"])

    def test_invalid_duplicate_missing_and_excessive_selections_never_guess_a_tensor(self):
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            for selection in ([], [UP_NAME, UP_NAME], [True], (UP_NAME,), [UP_NAME] * 257):
                with self.subTest(selection_type=type(selection).__name__):
                    self.assert_code("invalid_selection", lambda: diagnose(reference, quantized, selection))
            self.assert_code("gguf_tensor_missing", lambda: diagnose(reference, quantized, ["not-present"]))

    def test_work_limits_include_every_selected_element_and_precede_decoding(self):
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            for budget in (True, 0, -1, 1.5, DEFAULT_LIMITS.max_selected_elements + 1):
                self.assert_code("work_limit", lambda: diagnose(reference, quantized, [UP_NAME], max_elements=budget))
            with patch.object(reference, "iter_values", side_effect=AssertionError("No over-budget decoding")) as values:
                self.assert_code("work_limit", lambda: diagnose(reference, quantized, [UP_NAME, Q_NAME], max_elements=95))
                values.assert_not_called()
            self.assertEqual(diagnose(reference, quantized, [UP_NAME], max_elements=32)["selected_elements"], 32)
        with self.opened(pair, replace(DEFAULT_LIMITS, max_selected_elements=31)) as (reference, quantized):
            self.assert_code("work_limit", lambda: diagnose(reference, quantized, [UP_NAME], max_elements=32))

    def test_pair_metadata_shape_and_unquantized_identity_mismatches_fail_before_a_report(self):
        variants = [
            {"quantized_meta": replace_metadata(identity_metadata(quantized=True), "tokenizer.ggml.tokens", 9, (8, ["different", "tokens"]))},
            {"quantized": [*quantized_tensors()[:2], Tensor(NORM_NAME, (2,), F32, struct.pack("<2f", 2.0, -1.0))]},
            {"quantized": [Tensor(Q_NAME, (32, 1), Q4_0, Q4_POSITIVE), *quantized_tensors()[1:]]},
        ]
        for variant in variants:
            pair = self.pair(**variant)
            with self.opened(pair) as (reference, quantized):
                self.assert_code("repair_pair_mismatch", lambda: diagnose(reference, quantized, [Q_NAME]))

    def test_nonfinite_selected_reference_or_quantized_scale_is_never_a_zero_error(self):
        reference = reference_tensors()
        reference[0] = Tensor(Q_NAME, (32, 2), F16, struct.pack("<64e", math.inf, *([0.0] * 63)))
        quantized = quantized_tensors()
        quantized[0] = Tensor(Q_NAME, (32, 2), Q4_0, bytes.fromhex("00 7c") + Q4_POSITIVE[2:] + Q4_POSITIVE)
        for variant in ({"reference": reference}, {"quantized": quantized}):
            pair = self.pair(**variant)
            with self.opened(pair) as (left, right):
                self.assert_code("gguf_nonfinite_weight", lambda: diagnose(left, right, [Q_NAME]))

    def test_one_hot_probes_have_independently_known_row_outputs_and_distortions(self):
        pair = self.pair()
        vectors = [[1.0] + [0.0] * 31, [0.0] * 16 + [1.0] + [0.0] * 15]
        with self.opened(pair) as (reference, quantized):
            result = probe(reference, quantized, probe_document(vectors=vectors))
        self.assertEqual(result["paired_work_elements"], 128)
        self.assertEqual([row["index"] for row in result["vectors"]], [0, 1])
        first, second = result["vectors"]
        # e0: reference=(0,8), candidate=(-8,4). e16: reference=(4,12), candidate=(0,0).
        self.assertEqual(first["elements"], 2)
        self.assertAlmostEqual(first["rmse"], math.sqrt(40))
        self.assertAlmostEqual(first["relative_l2"], math.sqrt(1.25))
        self.assertEqual(first["max_absolute_error"], 8)
        self.assertAlmostEqual(second["rmse"], math.sqrt(80))
        self.assertEqual(second["relative_l2"], 1)
        self.assertEqual(second["max_absolute_error"], 12)

    def test_activation_choice_can_hide_or_expose_the_same_weight_error_without_behavior_claims(self):
        pair = self.pair()
        document = probe_document(tensor=UP_NAME, vectors=[[0.0] * 32, [1.0] + [0.0] * 31])
        with self.opened(pair) as (reference, quantized):
            result = probe(reference, quantized, document)
        self.assertEqual(result["vectors"][0]["rmse"], 0)
        self.assertIsNone(result["vectors"][0]["relative_l2"])
        self.assertEqual(result["vectors"][1]["rmse"], 1.25)
        self.assertEqual(result["evidence"], "supplied_linear_probes_only")
        self.assertEqual(result["behavioral_validation"], "not_run")

    def test_probe_fingerprint_binds_the_supplied_intent_and_provenance_is_detached(self):
        pair = self.pair()
        document = probe_document()
        before = json.loads(json.dumps(document))
        expected = hashlib.sha256((json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
        with self.opened(pair) as (reference, quantized):
            result = probe(reference, quantized, document)
            changed = json.loads(json.dumps(document)); changed["provenance"]["capture"] = "another authored capture"
            other = probe(reference, quantized, changed)
        self.assertEqual(result["canonical_probe_sha256"], expected)
        self.assertNotEqual(other["canonical_probe_sha256"], expected)
        result["provenance"]["capture"] = "consumer mutation"
        self.assertEqual(document, before)
        self.assertNotIn("vectors", result["provenance"])
        self.assertNotIn(str(self.directory), json.dumps(result))
        captured = probe_document()
        captured_before = json.loads(json.dumps(captured))
        expected_capture_hash = hashlib.sha256((json.dumps(captured_before, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
        with self.opened(pair) as (reference, quantized):
            original_values = reference.iter_values
            def mutate_caller_after_capture(tensor):
                captured["provenance"]["capture"] = "late caller mutation"
                captured["vectors"][0][0] = 0.0
                return original_values(tensor)
            with patch.object(reference, "iter_values", side_effect=mutate_caller_after_capture):
                stable = probe(reference, quantized, captured)
        self.assertEqual(stable["canonical_probe_sha256"], expected_capture_hash)
        self.assertEqual(stable["provenance"], captured_before["provenance"])
        self.assertAlmostEqual(stable["vectors"][0]["rmse"], math.sqrt(40))

    def test_probe_records_require_exact_version_kind_and_all_four_capture_fields(self):
        pair = self.pair()
        variants = [None, {**probe_document(), "extra": True}, {**probe_document(), "schema_version": True},
                    {**probe_document(), "kind": "unversioned"}, {**probe_document(), "provenance": {"capture": "missing bindings"}},
                    {**probe_document(), "provenance": {**probe_document()["provenance"], "capture": " "}}]
        with self.opened(pair) as (reference, quantized):
            for value in variants:
                self.assert_code("invalid_probes", lambda: probe(reference, quantized, value))

    def test_probe_vector_and_rank_bounds_reject_ragged_empty_boolean_or_oversized_values(self):
        pair = self.pair()
        variants = [[], [[1.0] * 31], [[1.0] * 16385], [[0.0] * 32] * 17, [[True] * 32]]
        with self.opened(pair) as (reference, quantized):
            for vectors in variants:
                self.assert_code("invalid_probes", lambda: probe(reference, quantized, probe_document(vectors=vectors)))
            for vectors in ([[math.nan] * 32], [[math.inf] * 32], [[10**10_000] * 32]):
                self.assert_code("invalid_json", lambda: probe(reference, quantized, probe_document(vectors=vectors)))
            self.assert_code("invalid_probes", lambda: probe(reference, quantized, probe_document(tensor=NORM_NAME, vectors=[[1.0, 0.0]])))

    def test_probe_work_budget_counts_all_vector_weight_pairs_before_forward_work(self):
        pair = self.pair()
        document = probe_document(vectors=[[1.0] * 32, [0.0] * 32])
        with self.opened(pair) as (reference, quantized):
            with patch.object(reference, "iter_values", side_effect=AssertionError("No over-budget forward work")) as values:
                self.assert_code("work_limit", lambda: probe(reference, quantized, document, max_elements=127))
                values.assert_not_called()
            self.assertEqual(probe(reference, quantized, document, max_elements=128)["paired_work_elements"], 128)

    def test_extreme_representable_metrics_remain_finite_and_true_product_overflow_fails(self):
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            for scale in (1e200, 1e-200):
                result = probe(reference, quantized, probe_document(tensor=UP_NAME, vectors=[[scale] + [0.0] * 31]))
                row = result["vectors"][0]
                self.assertTrue(math.isclose(row["rmse"], 1.25 * scale, rel_tol=1e-14, abs_tol=0.0))
                self.assertGreater(row["reference_rms"], 0.0)
                self.assertEqual(row["relative_l2"], 1.0)
                self.assertIsNone(row["relative_l2_unavailable"])
            with self.assertRaises(QuantopsyError) as error:
                probe(reference, quantized, probe_document(vectors=[[1e308] * 32]))
        self.assertIn(error.exception.code, ("numerical_overflow", "nonfinite_values"))
        self.assertNotIn("1e308", str(error.exception))
        mixed = self.pair(
            reference=[Tensor(UP_NAME, (32, 1), F16, struct.pack("<32e", 1.0, 1.0, 1.0, *([0.0] * 29)))],
            quantized=[Tensor(UP_NAME, (32, 1), Q4_0, bytes.fromhex("00 3c 89") + bytes.fromhex("88 " * 15))],
        )
        with self.opened(mixed) as (reference, quantized):
            cancellation = probe_document(tensor=UP_NAME, vectors=[[1e150, -1e150, 1e-150] + [0.0] * 29])
            row = probe(reference, quantized, cancellation)["vectors"][0]
            self.assertTrue(math.isclose(row["rmse"], 1e150, rel_tol=1e-14))
            self.assertTrue(math.isclose(row["reference_rms"], 1e-150, rel_tol=1e-14, abs_tol=0.0))
            self.assertTrue(math.isclose(row["relative_l2"], 1e300, rel_tol=1e-14))
            unrepresentable = probe_document(tensor=UP_NAME, vectors=[[1e150, -1e150, 1e-200] + [0.0] * 29])
            self.assert_code("numerical_overflow", lambda: probe(reference, quantized, unrepresentable))
        basis = self.pair(
            reference=[Tensor(Q_NAME, (32, 2), F16, struct.pack("<64e", *([1.0] + [0.0] * 31 + [0.0, 1.0] + [0.0] * 30)))],
            quantized=[Tensor(Q_NAME, (32, 2), Q4_0, bytes.fromhex("00 3c 89") + bytes.fromhex("88 " * 15) + bytes.fromhex("00 00") + bytes.fromhex("ff " * 16))],
        )
        with self.opened(basis) as (reference, quantized):
            # Actual outputs (1e150,1e-200) versus (1e150,0) have a nonzero
            # relative L2 below binary64 range, not a perfect match.
            underflow = probe_document(vectors=[[1e150, 1e-200] + [0.0] * 30])
            self.assert_code("numerical_overflow", lambda: probe(reference, quantized, underflow))
        smallest = self.pair(
            reference=[Tensor(UP_NAME, (32, 5), F16, struct.pack("<160e", *([1.0] + [0.0] * 159)))],
            quantized=[Tensor(UP_NAME, (32, 5), Q4_0, (bytes.fromhex("00 00") + bytes.fromhex("ff " * 16)) * 5)],
        )
        with self.opened(smallest) as (reference, quantized):
            # A single minimum-subnormal output among five rows has an
            # unrepresentable nonzero RMS; it must not become fabricated zero.
            underflow = probe_document(tensor=UP_NAME, vectors=[[5e-324] + [0.0] * 31])
            self.assert_code("numerical_overflow", lambda: probe(reference, quantized, underflow))

    def test_source_changes_before_or_during_streaming_prevent_a_completed_diagnosis(self):
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            pair.quantized.write_bytes(pair.quantized_bytes[:-1] + bytes([pair.quantized_bytes[-1] ^ 1]))
            self.assert_code("gguf_source_changed", lambda: diagnose(reference, quantized, [Q_NAME]))
        pair = self.pair()
        with self.opened(pair) as (reference, quantized):
            original = reference.iter_values
            def changing_values(tensor):
                for index, value in enumerate(original(tensor)):
                    if index == 1:
                        pair.quantized.write_bytes(pair.quantized_bytes[:-1] + bytes([pair.quantized_bytes[-1] ^ 1]))
                    yield value
            with patch.object(reference, "iter_values", side_effect=changing_values):
                self.assert_code("gguf_source_changed", lambda: diagnose(reference, quantized, [Q_NAME]))
