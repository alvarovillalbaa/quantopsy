"""Tier 2 data-only behavior comparison and CLI/JSON seams, without inference."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from decimal import localcontext
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from quantopsy.behavior import compare_behavior
from quantopsy.cli import main
from quantopsy.errors import QuantopsyError
from quantopsy.io import read_json_document
from tests.behavior_fixtures import ROLES, authored_documents, bind_manifest_plan, bind_records, digest, encoded


class BehaviorComparisonTier2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="quantopsy-authored-behavior-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_documents(self, documents, raw=None):
        paths = {}
        for name, value in documents.items():
            paths[name] = self.root / (name + ".json")
            paths[name].write_bytes(raw[name] if raw and name in raw else encoded(value))
        return paths

    def compare_paths(self, paths, *, manifest=True):
        return compare_behavior(paths["suite"], paths["config"], paths["models"],
                                paths["reference"], paths["quantized"], paths["candidate"],
                                manifest_path=paths["manifest"] if manifest else None)

    def compare(self, documents=None, *, manifest=True, raw=None):
        paths = self.write_documents(authored_documents() if documents is None else documents, raw)
        return self.compare_paths(paths, manifest=manifest)

    def refused(self, documents, code=None, *, manifest=True, raw=None):
        with self.assertRaises(QuantopsyError) as caught:
            self.compare(documents, manifest=manifest, raw=raw)
        if code is not None:
            self.assertEqual(caught.exception.code, code)

    def case(self, report, task=0, metric=0):
        return report["tasks"][task]["cases"][metric]

    def column(self, documents, role, metric, value):
        for task in documents[role]["tasks"]:
            task["metrics"][metric]["value"] = value

    def command(self, paths, output, *, manifest=True):
        arguments = ["compare-behavior", "--suite", paths["suite"], "--config", paths["config"],
                     "--models", paths["models"], "--reference-run", paths["reference"],
                     "--quantized-run", paths["quantized"], "--candidate-run", paths["candidate"],
                     "--output", output]
        if manifest:
            arguments += ["--repair-manifest", paths["manifest"]]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(list(map(str, arguments)))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_directed_metrics_accept_exact_recovery_and_regression_boundaries(self):
        report = self.compare()
        self.assertEqual(report["acceptance"], {
            "complete_runs": True, "qualifying_loss_cases": 4, "recovered_cases": 4,
            "regression_cases": 0, "measurements_pass": True,
            "repair_manifest_records_bound": True, "supplied_recovery_records_pass": True})
        lower, higher = self.case(report), self.case(report, metric=1)
        self.assertEqual((lower["quantization_loss"], lower["candidate_improvement"],
                          lower["recovery_fraction"], lower["reference_regression"]),
                         ("2", "1", "0.5", "1"))
        self.assertEqual((higher["quantization_loss"], higher["candidate_improvement"],
                          higher["recovery_fraction"], higher["reference_regression"]),
                         ("0.4", "0.2", "0.5", "0.2"))

    def test_report_binds_raw_input_bytes_and_states_unauthenticated_evidence_limits(self):
        documents = authored_documents()
        raw_models = json.dumps(documents["models"], separators=(",", ":")).encode()
        report = self.compare(documents, raw={"models": raw_models})
        self.assertEqual(report["inputs"], {
            "suite_sha256": digest(encoded(documents["suite"])),
            "config_sha256": digest(encoded(documents["config"])),
            "models_sha256": digest(raw_models),
            "run_sha256": {role: digest(encoded(documents[role])) for role in ROLES},
            "repair_manifest_sha256": digest(encoded(documents["manifest"]))})
        self.assertEqual(report["evidence"], "caller_supplied_measurements_identity_checked")
        self.assertEqual(report["inference_execution"], "not_performed_by_comparator")
        self.assertEqual(report["model_payloads"], "not_opened_by_comparator")
        self.assertEqual(report["freeze_chronology"], "not_established_by_imported_records")
        self.assertEqual(report["source_execution_authenticity"], "not_established_by_imported_records")
        self.assertEqual(report["behavioral_recovery"], "requires_independently_verified_inference_evidence")
        self.assertNotIn(str(self.root), json.dumps(report))

    def test_good_measurements_without_repair_manifest_cannot_pass_bound_recovery_records(self):
        report = self.compare(manifest=False)
        self.assertTrue(report["acceptance"]["measurements_pass"])
        self.assertFalse(report["acceptance"]["repair_manifest_records_bound"])
        self.assertFalse(report["acceptance"]["supplied_recovery_records_pass"])
        self.assertIsNone(report["inputs"]["repair_manifest_sha256"])

    def test_decimal_product_decisions_remain_exact_below_float_resolution(self):
        documents = authored_documents()
        documents["suite"]["metrics"][0].update(
            min_quantization_loss="0.000000000000000000000003", max_reference_regression="1")
        self.column(documents, "reference", 0, "1.000000000000000000000000")
        self.column(documents, "quantized", 0, "1.000000000000000000000003")
        self.column(documents, "candidate", 0, "1.000000000000000000000002")
        bind_records(documents)
        with localcontext() as context:
            context.prec = 3
            report = self.compare(documents)
        case = self.case(report)
        self.assertEqual(case["quantization_loss"], "0.000000000000000000000003")
        self.assertEqual(case["candidate_improvement"], "0.000000000000000000000001")
        self.assertTrue(case["qualifying_loss"])
        self.assertFalse(case["recovered"])
        self.assertFalse(report["acceptance"]["measurements_pass"])
        self.column(documents, "candidate", 0, "1.000000000000000000000001")
        self.assertTrue(self.compare(documents)["acceptance"]["measurements_pass"])

    def test_loss_below_frozen_minimum_is_not_qualifying(self):
        documents = authored_documents()
        documents["suite"]["metrics"][0]["min_quantization_loss"] = "2.000000000000000000000001"
        bind_records(documents)
        report = self.compare(documents)
        self.assertFalse(self.case(report)["qualifying_loss"])
        self.assertFalse(self.case(report)["recovered"])
        self.assertEqual(report["acceptance"]["qualifying_loss_cases"], 2)

    def test_zero_quantization_loss_cannot_vacuously_pass(self):
        documents = authored_documents()
        for role in ("quantized", "candidate"):
            documents[role]["tasks"] = deepcopy(documents["reference"]["tasks"])
        for metric in documents["suite"]["metrics"]:
            metric["min_quantization_loss"] = "0"
            metric["min_recovery_fraction"] = "0"
        bind_records(documents)
        report = self.compare(documents)
        self.assertEqual(report["acceptance"]["qualifying_loss_cases"], 0)
        self.assertFalse(report["acceptance"]["measurements_pass"])
        self.assertIsNone(self.case(report)["recovery_fraction"])

    def test_zero_recovery_threshold_still_requires_actual_candidate_improvement(self):
        documents = authored_documents()
        documents["candidate"]["tasks"] = deepcopy(documents["quantized"]["tasks"])
        for metric in documents["suite"]["metrics"]:
            metric.update(min_recovery_fraction="0", max_reference_regression="9")
        bind_records(documents)
        report = self.compare(documents)
        self.assertEqual(report["acceptance"]["qualifying_loss_cases"], 4)
        self.assertEqual(report["acceptance"]["recovered_cases"], 0)
        self.assertEqual(self.case(report)["candidate_improvement"], "0")
        self.assertFalse(report["acceptance"]["measurements_pass"])

    def test_reference_regression_can_block_otherwise_recovered_case(self):
        documents = authored_documents()
        documents["suite"]["metrics"][0]["max_reference_regression"] = "0.999999999999999999999999"
        bind_records(documents)
        report = self.compare(documents)
        self.assertTrue(self.case(report)["recovered"])
        self.assertTrue(self.case(report)["reference_regression_exceeded"])
        self.assertEqual(report["acceptance"]["regression_cases"], 2)
        self.assertFalse(report["acceptance"]["measurements_pass"])

    def test_quantized_regression_is_checked_even_when_quantization_improved_reference(self):
        documents = authored_documents()
        self.column(documents, "reference", 0, "5")
        self.column(documents, "quantized", 0, "3")
        self.column(documents, "candidate", 0, "4")
        report = self.compare(documents)
        case = self.case(report)
        self.assertEqual(case["quantization_loss"], "-2")
        self.assertFalse(case["qualifying_loss"])
        self.assertEqual(case["quantized_regression"], "1")
        self.assertTrue(case["quantized_regression_exceeded"])
        self.assertFalse(case["reference_regression_exceeded"])
        self.assertFalse(report["acceptance"]["measurements_pass"])

    def test_one_unrecovered_task_metric_blocks_aggregate_acceptance(self):
        documents = authored_documents()
        documents["candidate"]["tasks"][1]["metrics"][1]["value"] = "0.5"
        report = self.compare(documents)
        self.assertEqual(report["acceptance"]["qualifying_loss_cases"], 4)
        self.assertEqual(report["acceptance"]["recovered_cases"], 3)
        self.assertFalse(self.case(report, task=1, metric=1)["recovered"])
        self.assertFalse(report["acceptance"]["measurements_pass"])

    def test_a_case_exceeding_both_regression_limits_is_counted_once(self):
        documents = authored_documents()
        documents["candidate"]["tasks"][0]["metrics"][0]["value"] = "5"
        report = self.compare(documents)
        case = self.case(report)
        self.assertTrue(case["reference_regression_exceeded"])
        self.assertTrue(case["quantized_regression_exceeded"])
        self.assertEqual(report["acceptance"]["regression_cases"], 1)

    def test_incomplete_ordered_prefix_retains_original_error_and_missing_cells(self):
        documents = authored_documents()
        documents["candidate"].update(status="incomplete", error="authored interruption after task-a")
        documents["candidate"]["tasks"] = documents["candidate"]["tasks"][:1]
        report = self.compare(documents)
        self.assertEqual(report["runs"]["candidate"], {
            "status": "incomplete", "error": "authored interruption after task-a", "measured_tasks": 1})
        self.assertEqual(self.case(report)["status"], "compared")
        missing = self.case(report, task=1)
        self.assertEqual(missing["status"], "unavailable")
        self.assertEqual(missing["values"], {"reference": "2", "quantized": "4", "candidate": None})
        self.assertIsNone(missing["candidate_improvement"])
        self.assertFalse(missing["recovered"])
        self.assertFalse(report["acceptance"]["complete_runs"])
        self.assertFalse(report["acceptance"]["supplied_recovery_records_pass"])

    def test_error_with_no_completed_tasks_is_reported_without_synthetic_measurements(self):
        documents = authored_documents()
        documents["reference"].update(status="error", error="authored start refusal", tasks=[])
        report = self.compare(documents)
        self.assertEqual(report["runs"]["reference"]["status"], "error")
        self.assertEqual(report["runs"]["reference"]["error"], "authored start refusal")
        self.assertEqual(report["runs"]["reference"]["measured_tasks"], 0)
        self.assertTrue(all(case["status"] == "unavailable" for task in report["tasks"] for case in task["cases"]))
        self.assertEqual(report["acceptance"]["qualifying_loss_cases"], 0)
        self.assertFalse(report["acceptance"]["measurements_pass"])

    def test_complete_coverage_and_all_run_prefixes_require_exact_task_order(self):
        variants = []
        for status, rows in (("complete", [0]), ("complete", [1, 0]),
                             ("complete", [0, 0]), ("incomplete", [1]), ("error", [1])):
            documents = authored_documents()
            original = documents["candidate"]["tasks"]
            documents["candidate"].update(status=status, error=None if status == "complete" else "authored stop",
                                          tasks=[deepcopy(original[index]) for index in rows])
            variants.append(documents)
        for documents in variants:
            with self.subTest(status=documents["candidate"]["status"], rows=documents["candidate"]["tasks"]):
                self.refused(documents, "behavior_coverage_mismatch")

    def test_metric_coverage_and_order_are_frozen_per_task(self):
        for indexes in ([0], [1, 0], [0, 0], [0, 1, 0]):
            documents = authored_documents()
            original = documents["candidate"]["tasks"][0]["metrics"]
            documents["candidate"]["tasks"][0]["metrics"] = [deepcopy(original[index]) for index in indexes]
            with self.subTest(indexes=indexes):
                self.refused(documents, "behavior_coverage_mismatch")

    def test_role_labels_cannot_be_swapped(self):
        for role in ROLES:
            documents = authored_documents()
            documents[role]["role"] = "candidate" if role != "candidate" else "reference"
            with self.subTest(role=role):
                self.refused(documents, "behavior_identity_mismatch")

    def test_engine_name_revision_and_source_hash_must_match_configuration(self):
        for field, value in (("name", "another-engine"), ("revision", "different-r2"), ("source_sha256", "9" * 64)):
            documents = authored_documents()
            documents["candidate"]["engine"][field] = value
            with self.subTest(field=field):
                self.refused(documents, "behavior_identity_mismatch")

    def test_configuration_requires_exact_bound_bytes_in_suite_and_every_run(self):
        documents = authored_documents()
        config_bytes = json.dumps(documents["config"], separators=(",", ":")).encode()
        self.refused(documents, "behavior_identity_mismatch", raw={"config": config_bytes})
        documents["suite"]["config_sha256"] = digest(config_bytes)
        suite_hash = digest(encoded(documents["suite"]))
        for role in ROLES:
            documents[role]["suite_sha256"] = suite_hash
            documents[role]["config_sha256"] = digest(config_bytes)
        self.assertTrue(self.compare(documents, raw={"config": config_bytes})["acceptance"]["measurements_pass"])
        for role in ROLES:
            changed = deepcopy(documents)
            changed[role]["config_sha256"] = "0" * 64
            with self.subTest(role=role):
                self.refused(changed, "behavior_identity_mismatch", raw={"config": config_bytes})

    def test_model_name_size_and_complete_hash_are_bound_for_each_role(self):
        for role in ROLES:
            for field, value in (("name", "another.gguf"), ("size", 1111), ("sha256", "9" * 64)):
                documents = authored_documents()
                documents[role]["model"][field] = value
                with self.subTest(role=role, field=field):
                    self.refused(documents, "behavior_identity_mismatch")

    def test_suite_hash_changes_with_whitespace_and_requires_every_run_to_rebind(self):
        documents = authored_documents()
        suite_bytes = b" \n" + encoded(documents["suite"])
        self.refused(documents, "behavior_identity_mismatch", raw={"suite": suite_bytes})
        for role in ROLES:
            documents[role]["suite_sha256"] = digest(suite_bytes)
        report = self.compare(documents, raw={"suite": suite_bytes})
        self.assertEqual(report["inputs"]["suite_sha256"], digest(suite_bytes))
        self.assertTrue(report["acceptance"]["measurements_pass"])

    def test_schema_extra_fields_are_rejected_at_nested_record_boundaries(self):
        paths = (("suite",), ("suite", "metrics", 0), ("suite", "tasks", 0),
                 ("config",), ("config", "engine"), ("models",), ("models", "candidate"),
                 ("candidate",), ("candidate", "tasks", 0), ("candidate", "tasks", 0, "metrics", 0))
        for path in paths:
            documents = authored_documents()
            record = documents
            for key in path:
                record = record[key]
            record["unsupported"] = "authored extra"
            bind_records(documents)
            with self.subTest(path=path):
                self.refused(documents, "behavior_invalid_record")

    def test_duplicate_frozen_tasks_and_metrics_are_rejected(self):
        for field in ("tasks", "metrics"):
            documents = authored_documents()
            documents["suite"][field][1]["id"] = documents["suite"][field][0]["id"]
            bind_records(documents)
            with self.subTest(field=field):
                self.refused(documents, "behavior_invalid_suite")

    def test_measurements_reject_nondecimal_strings_numeric_values_and_booleans(self):
        invalid = (True, False, 1, 0.5, None, "NaN", "Infinity", "1e-3", ".5", "01", "+1",
                   " 1", "1 ", "1.", "0.1234567890123456789012345", "1000000000000000000")
        for value in invalid:
            documents = authored_documents()
            documents["candidate"]["tasks"][0]["metrics"][0]["value"] = value
            with self.subTest(value=value):
                self.refused(documents, "behavior_invalid_number")

    def test_thresholds_reject_negative_limits_and_out_of_range_recovery_fractions(self):
        invalid = (("min_quantization_loss", "-0.1"), ("max_reference_regression", "-1"),
                   ("max_quantized_regression", "-1"), ("min_recovery_fraction", "-0.1"),
                   ("min_recovery_fraction", "1.000000000000000000000001"),
                   ("min_recovery_fraction", True))
        for field, value in invalid:
            documents = authored_documents()
            documents["suite"]["metrics"][0][field] = value
            bind_records(documents)
            with self.subTest(field=field, value=value):
                self.refused(documents, "behavior_invalid_number")

    def test_integer_flags_and_status_fields_reject_bool_coercion_and_inconsistent_errors(self):
        variants = []
        for key, value in (("schema_version", True), ("context_tokens", True), ("batch_tokens", 257),
                           ("threads", 0), ("seed", 2**31), ("add_bos", 1), ("add_eos", "false")):
            documents = authored_documents()
            documents["config"][key] = value
            bind_records(documents)
            variants.append(documents)
        for status, error in (("complete", "authored error"), ("error", None),
                              ("incomplete", ""), (True, "authored"), ("unknown", None)):
            documents = authored_documents()
            documents["candidate"].update(status=status, error=error)
            variants.append(documents)
        for index, documents in enumerate(variants):
            with self.subTest(index=index):
                self.refused(documents)

    def test_suite_bounds_identifiers_and_text_are_strict(self):
        variants = []
        for field, value in (("tasks", []), ("metrics", []),
                             ("tasks", [{"id": "task-" + str(index), "prompt": "", "target": "x"} for index in range(129)]),
                             ("metrics", [dict(authored_documents()["suite"]["metrics"][0], id="metric-" + str(index)) for index in range(9)]),
                             ("suite_id", "../unsafe")):
            documents = authored_documents()
            documents["suite"][field] = value
            variants.append(documents)
        for field, value in (("target", ""), ("target", "x" * 16_385), ("prompt", "\0"), ("id", "task with spaces")):
            documents = authored_documents()
            documents["suite"]["tasks"][0][field] = value
            variants.append(documents)
        for index, documents in enumerate(variants):
            bind_records(documents)
            with self.subTest(index=index):
                self.refused(documents)

    def test_both_supported_repair_manifest_versions_bind_supplied_records(self):
        documents = authored_documents()
        for version, kind, reference_bytes, added in ((1, 1, 128, 92), (2, 0, 256, 220)):
            manifest = documents["manifest"]
            manifest.update(schema_version=version, added_weight_bytes=added, max_added_bytes=added)
            manifest["selected"][0].update(reference_kind=kind, reference_bytes=reference_bytes)
            bind_manifest_plan(documents)
            with self.subTest(version=version):
                report = self.compare(documents)
                self.assertTrue(report["acceptance"]["supplied_recovery_records_pass"])
                self.assertEqual(report["inputs"]["repair_manifest_sha256"], digest(encoded(manifest)))

    def test_manifest_full_reference_quantized_and_candidate_identities_must_match(self):
        for role in ROLES:
            for field, value in (("name", "changed.gguf"), ("size", 1234), ("sha256", "9" * 64)):
                documents = authored_documents()
                manifest_role = "output" if role == "candidate" else role
                documents["manifest"][manifest_role][field] = value
                bind_manifest_plan(documents)
                with self.subTest(role=role, field=field):
                    self.refused(documents, "behavior_identity_mismatch")

    def test_manifest_cannot_bind_an_arbitrary_or_stale_canonical_plan_hash(self):
        documents = authored_documents()
        documents["manifest"]["plan_sha256"] = "0" * 64
        self.refused(documents, "behavior_invalid_manifest")
        documents = authored_documents()
        documents["manifest"]["provenance"]["revision"] = "authored-r2-with-stale-hash"
        self.refused(documents, "behavior_invalid_manifest")
        bind_manifest_plan(documents)
        self.assertTrue(self.compare(documents)["acceptance"]["supplied_recovery_records_pass"])

    def test_manifest_requires_strict_plan_arithmetic_verification_and_unclaimed_behavior(self):
        variants = []
        for field, value in (("schema_version", True), ("plan_sha256", "A" * 64),
                             ("metadata_omissions", ["unrelated.key"]), ("behavioral_validation", "passed"),
                             ("added_weight_bytes", 91), ("reserve_bytes", True)):
            documents = authored_documents()
            documents["manifest"][field] = value
            variants.append(documents)
        for flag in authored_documents()["manifest"]["verification"]:
            for value in (False, 1):
                documents = authored_documents()
                documents["manifest"]["verification"][flag] = value
                variants.append(documents)
        for path in (("manifest",), ("manifest", "verification"), ("manifest", "selected", 0)):
            documents = authored_documents()
            record = documents
            for key in path:
                record = record[key]
            record["extra"] = True
            variants.append(documents)
        for index, documents in enumerate(variants):
            with self.subTest(index=index):
                self.refused(documents)

    def test_json_document_hash_is_of_the_exact_single_parsed_byte_sequence(self):
        path = self.root / "raw.json"
        raw = b' \n{ "label" : "caf\xc3\xa9", "value":  1 }\n'
        path.write_bytes(raw)
        value, actual_hash = read_json_document(path)
        self.assertEqual(value, {"label": "café", "value": 1})
        self.assertEqual(actual_hash, digest(raw))
        self.assertNotEqual(actual_hash, digest(encoded(value)))

    def test_json_changed_during_read_and_duplicate_nonfinite_inputs_are_refused(self):
        path = self.root / "changing.json"
        replacement = self.root / "replacement.json"
        path.write_bytes(b'{"value":1}')
        replacement.write_bytes(b'{"value":2}')
        real_lstat = os.lstat
        calls = []

        def replace_before_final_identity_check(target):
            if Path(target) == path:
                calls.append(True)
                if len(calls) == 2:
                    os.replace(replacement, path)
            return real_lstat(target)

        with patch("quantopsy.io.os.lstat", side_effect=replace_before_final_identity_check):
            with self.assertRaises(QuantopsyError) as caught:
                read_json_document(path)
        self.assertEqual(caught.exception.code, "invalid_json_file")
        self.assertEqual(path.read_bytes(), b'{"value":2}')
        for raw in (b'{"value":1,"value":2}', b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e400}'):
            path.write_bytes(raw)
            with self.subTest(raw=raw):
                with self.assertRaises(QuantopsyError) as caught:
                    read_json_document(path)
                self.assertEqual(caught.exception.code, "invalid_json")

    def test_cli_creates_comparison_without_opening_models_or_entering_gguf_profile(self):
        paths = self.write_documents(authored_documents())
        output = self.root / "comparison.json"
        with patch("quantopsy.cli.GGUFFile", side_effect=AssertionError("model payload must remain unopened")), \
             patch("quantopsy.cli.profile_version", side_effect=AssertionError("compare must branch before GGUF profile")):
            status, stdout, stderr = self.command(paths, output)
        self.assertEqual((status, stderr), (0, ""))
        self.assertEqual(json.loads(stdout), {"saved_file": "comparison.json", "status": "created"})
        report = json.loads(output.read_bytes())
        self.assertEqual(report["kind"], "quantopsy.behavior-comparison")
        self.assertTrue(report["acceptance"]["supplied_recovery_records_pass"])
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_cli_create_only_output_refuses_file_and_symlink_without_overwrite(self):
        paths = self.write_documents(authored_documents())
        existing = self.root / "existing.json"
        existing.write_bytes(b"authored existing record\n")
        linked = self.root / "linked.json"
        linked.symlink_to(existing)
        for destination in (existing, linked):
            with self.subTest(destination=destination.name):
                status, stdout, stderr = self.command(paths, destination)
                self.assertEqual((status, stdout), (2, ""))
                self.assertEqual(json.loads(stderr)["error"]["code"], "io_error")
                self.assertNotIn(str(self.root), stderr)
                self.assertEqual(existing.read_bytes(), b"authored existing record\n")
                self.assertTrue(linked.is_symlink())

    def test_cli_invalid_bound_records_emit_finite_error_and_no_partial_report(self):
        documents = authored_documents()
        documents["candidate"]["role"] = "reference"
        paths = self.write_documents(documents)
        output = self.root / "refused.json"
        with patch("quantopsy.cli.GGUFFile", side_effect=AssertionError("models must stay unopened")):
            status, stdout, stderr = self.command(paths, output)
        self.assertEqual((status, stdout), (2, ""))
        self.assertEqual(set(json.loads(stderr)), {"error"})
        self.assertEqual(json.loads(stderr)["error"]["code"], "behavior_identity_mismatch")
        self.assertNotIn(str(self.root), stderr)
        self.assertFalse(output.exists())
