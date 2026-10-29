"""Authored T2 CLI/JSON cases. UNRUN; only tiny owned local fixture files."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from quantopsy.cli import main
from quantopsy.errors import QuantopsyError
from quantopsy.gguf import GGUFFile
from quantopsy.io import canonical_json_bytes, read_json, write_json_create
from tests.fixtures import F16, Q_NAME, UP_NAME, provenance, sha256, write_pair


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="quantopsy-cli-")
        self.directory = Path(self.temporary.name).resolve()
        self.addCleanup(self.temporary.cleanup)
        self.pair = write_pair(self.directory)

    def command(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(list(map(str, arguments)))
        return status, stdout.getvalue(), stderr.getvalue()

    def supplied_json(self, name, value):
        target = self.directory / name
        target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return target

    def plan(self, name="plan.json"):
        declared = self.supplied_json("provenance-" + name, provenance())
        target = self.directory / name
        status, stdout, stderr = self.command("plan", self.pair.reference, self.pair.quantized,
            "--tensor", Q_NAME, "--provenance", declared, "--max-added-bytes", 92, "--output", target)
        self.assertEqual((status, stderr), (0, ""))
        return target, json.loads(stdout)

    def assert_finite_failure(self, arguments, private_text):
        status, stdout, stderr = self.command(*arguments)
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        payload = json.loads(stderr)
        self.assertEqual(set(payload), {"error"})
        self.assertEqual(set(payload["error"]), {"code", "message"})
        self.assertTrue(payload["error"]["code"])
        self.assertNotIn(private_text, stderr)
        return payload

    def test_inspect_prints_actual_container_hash_and_explicit_unrun_behavior(self):
        status, stdout, stderr = self.command("inspect", self.pair.quantized)
        self.assertEqual((status, stderr), (0, ""))
        result = json.loads(stdout)
        self.assertEqual(result["kind"], "quantopsy.container-inspection")
        self.assertEqual(result["file"]["sha256"], sha256(self.pair.quantized_bytes))
        self.assertEqual(result["architecture"], "llama")
        self.assertEqual(result["tensor_types"], {"Q4_0": 2, "F32": 1})
        self.assertEqual(result["behavioral_validation"], "not_run")
        self.assertNotIn(str(self.directory), stdout)

    def test_diagnose_command_uses_selected_actual_weights_without_a_runner(self):
        status, stdout, stderr = self.command("diagnose", self.pair.reference, self.pair.quantized, "--tensor", UP_NAME)
        self.assertEqual((status, stderr), (0, ""))
        result = json.loads(stdout)
        self.assertEqual(result["kind"], "quantopsy.weight-diagnosis")
        self.assertEqual(result["tensors"][0]["rmse"], 1.25)
        self.assertEqual(result["evidence"], "parameter_error_only")
        self.assertEqual(result["behavioral_validation"], "not_run")

    def test_probe_command_reads_explicit_calibration_and_reports_only_supplied_linear_evidence(self):
        probes = {"schema_version": 1, "kind": "quantopsy.linear-probes", "tensor": UP_NAME,
                  "vectors": [[1.0] + [0.0] * 31],
                  "provenance": {"model_id": "authored/tiny", "revision": "r1", "dataset": "authored", "capture": "manual independent vector"}}
        path = self.supplied_json("probes.json", probes)
        status, stdout, stderr = self.command("probe", self.pair.reference, self.pair.quantized, "--probes", path)
        self.assertEqual((status, stderr), (0, ""))
        result = json.loads(stdout)
        self.assertEqual(result["vectors"][0]["rmse"], 1.25)
        self.assertEqual(result["provenance"], probes["provenance"])
        self.assertEqual(result["paired_work_elements"], 32)
        self.assertEqual(result["evidence"], "supplied_linear_probes_only")
        self.assertEqual(result["behavioral_validation"], "not_run")

    def test_plan_command_creates_a_new_provenance_bound_record_with_exact_source_hashes(self):
        target, receipt = self.plan()
        self.assertEqual(receipt, {"saved_file": "plan.json", "status": "created"})
        result = json.loads(target.read_text())
        self.assertEqual(result["kind"], "quantopsy.repair-plan")
        self.assertEqual(result["reference"]["sha256"], sha256(self.pair.reference_bytes))
        self.assertEqual(result["quantized"]["sha256"], sha256(self.pair.quantized_bytes))
        self.assertEqual(result["provenance"], provenance())
        self.assertEqual(result["added_weight_bytes"], 92)
        self.assertEqual(result["behavioral_validation"], "not_run")
        self.assertNotIn(str(self.directory), target.read_text())

    def test_repair_command_writes_real_selected_f16_bytes_without_subprocess_or_network(self):
        plan, _receipt = self.plan()
        output = self.directory / "candidate-bundle"
        with patch("subprocess.Popen", side_effect=AssertionError("No inference process")) as process, \
                patch("socket.create_connection", side_effect=AssertionError("No network")) as network:
            status, stdout, stderr = self.command("repair", self.pair.reference, self.pair.quantized, "--plan", plan,
                                                   "--output-dir", output, "--reserve-bytes", 0)
        self.assertEqual((status, stderr), (0, ""))
        process.assert_not_called(); network.assert_not_called()
        result = json.loads(stdout)
        self.assertEqual(result, json.loads((output / "repair.json").read_text()))
        self.assertTrue(result["verification"]["container_reparsed"])
        self.assertEqual(result["behavioral_validation"], "not_run")
        with GGUFFile(output / "candidate.gguf") as candidate, GGUFFile(self.pair.reference) as reference, GGUFFile(self.pair.quantized) as quantized:
            self.assertEqual(candidate.tensor(Q_NAME).kind, F16)
            self.assertEqual(candidate.hash_tensor(candidate.tensor(Q_NAME)), reference.hash_tensor(reference.tensor(Q_NAME)))
            self.assertEqual(candidate.hash_tensor(candidate.tensor(UP_NAME)), quantized.hash_tensor(quantized.tensor(UP_NAME)))
        self.assertEqual(self.pair.reference.read_bytes(), self.pair.reference_bytes)
        self.assertEqual(self.pair.quantized.read_bytes(), self.pair.quantized_bytes)

    def test_cli_failures_return_finite_json_without_echoing_paths_or_argument_payloads(self):
        private = "private-input-marker"
        missing = self.directory / private
        self.assert_finite_failure(["inspect", missing], private)
        self.assert_finite_failure(["diagnose", self.pair.reference, self.pair.quantized, "--tensor", UP_NAME,
                                    "--max-elements", private], private)
        self.assert_finite_failure(["unknown-" + private], private)
        self.assert_finite_failure(["inspect", self.pair.quantized, "--output", ""], str(self.directory))

    def test_pair_open_failure_closes_prior_resources_through_the_actual_cli_stack(self):
        opened = []
        def track(path):
            source = GGUFFile(path)
            opened.append(source)
            return source
        bad = self.directory / "invalid.gguf"; bad.write_bytes(b"not a container")
        with patch("quantopsy.cli.GGUFFile", side_effect=track):
            self.assert_finite_failure(["diagnose", self.pair.reference, bad, "--tensor", UP_NAME], str(self.directory))
        self.assertEqual(len(opened), 1)
        with self.assertRaises(QuantopsyError) as error:
            opened[0].sha256()
        self.assertEqual(error.exception.code, "gguf_closed")

    def test_cli_report_output_refuses_existing_files_and_retains_their_exact_bytes(self):
        output = self.directory / "already.json"; output.write_bytes(b"authored prior record\n")
        self.assert_finite_failure(["inspect", self.pair.quantized, "--output", output], str(self.directory))
        self.assertEqual(output.read_bytes(), b"authored prior record\n")
        self.assertEqual(self.pair.quantized.read_bytes(), self.pair.quantized_bytes)

    def test_json_roundtrip_preserves_unicode_zero_and_canonical_key_order(self):
        value = {"z": [0, False, None, "café 🌍"], "a": {"empty": ""}}
        data = canonical_json_bytes(value)
        self.assertEqual(data, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        path = self.directory / "unicode.json"; path.write_bytes(data)
        self.assertEqual(read_json(path), value)
        self.assertTrue(data.endswith(b"\n"))

    def test_json_reads_reject_invalid_utf8_duplicates_surrogates_and_nonfinite_numbers(self):
        invalid = [b"\xff", b'{"duplicate":1,"duplicate":2}', b'{"value":"\\ud800"}',
                   b'{"\\udfff":1}', b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e400}', b'{unfinished']
        for index, data in enumerate(invalid):
            with self.subTest(index=index):
                path = self.directory / f"invalid-{index}.json"; path.write_bytes(data)
                with self.assertRaises(QuantopsyError) as error:
                    read_json(path)
                self.assertEqual(error.exception.code, "invalid_json")

    def test_json_depth_entry_byte_and_limit_budgets_reject_before_unbounded_output(self):
        value = 0
        for _ in range(18):
            value = [value]
        for oversized in (value, [None] * 100_001):
            with self.assertRaises(QuantopsyError) as error:
                canonical_json_bytes(oversized)
            self.assertEqual(error.exception.code, "invalid_json")
        path = self.directory / "bytes.json"; path.write_bytes(b'{"a":1}')
        self.assertEqual(read_json(path, max_bytes=7), {"a": 1})
        with self.assertRaises(QuantopsyError) as error:
            read_json(path, max_bytes=6)
        self.assertEqual(error.exception.code, "json_too_large")
        for limit in (True, 0, -1, 1.5):
            with self.assertRaises(QuantopsyError):
                canonical_json_bytes({}, max_bytes=limit)
        with self.assertRaises(QuantopsyError):
            canonical_json_bytes({"a": 1}, max_bytes=1)

    def test_json_output_rejects_cycles_non_json_values_and_isolated_unicode_units(self):
        cycle = []; cycle.append(cycle)
        for value in (cycle, {"a": b"bytes"}, {"a": object()}, {1: "numeric key"}, {"a": "\ud800"}, {"a": float("nan")}, {"a": float("inf")}):
            with self.assertRaises(QuantopsyError) as error:
                canonical_json_bytes(value)
            self.assertEqual(error.exception.code, "invalid_json")

    def test_json_create_is_private_and_input_output_paths_refuse_symlinks_or_nonfiles(self):
        target = self.directory / "created.json"; write_json_create(target, {"authored": True})
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        before = target.read_bytes()
        with self.assertRaises(OSError):
            write_json_create(target, {"replacement": True})
        link = self.directory / "link.json"; link.symlink_to(target)
        with self.assertRaises(OSError):
            write_json_create(link, {"replacement": True})
        with self.assertRaises(QuantopsyError):
            read_json(link)
        with self.assertRaises(QuantopsyError):
            read_json(self.directory)
        fifo = self.directory / "input.pipe"; os.mkfifo(fifo, 0o600)
        with self.assertRaises(QuantopsyError):
            read_json(fifo)
        self.assertEqual(target.read_bytes(), before)
        self.assertTrue(link.is_symlink())

    def test_failed_json_fsync_removes_only_the_owned_partial_and_closes_its_descriptor(self):
        target = self.directory / "failed.json"; descriptors = []; original = os.open
        def capture(*args, **kwargs):
            descriptor = original(*args, **kwargs); descriptors.append(descriptor); return descriptor
        with patch("quantopsy.io.os.open", side_effect=capture), patch("quantopsy.io.os.fsync", side_effect=OSError("private fsync details")):
            with self.assertRaises(OSError):
                write_json_create(target, {"authored": True})
        self.assertFalse(target.exists())
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.assertEqual(self.pair.reference.read_bytes(), self.pair.reference_bytes)

    def test_json_descriptor_setup_failures_close_the_fd_and_never_claim_a_completed_record(self):
        original_open = os.open
        for stage in ("fstat", "fdopen"):
            with self.subTest(stage=stage):
                target = self.directory / (stage + ".json"); descriptors = []
                def capture(*args, **kwargs):
                    descriptor = original_open(*args, **kwargs); descriptors.append(descriptor); return descriptor
                with patch("quantopsy.io.os.open", side_effect=capture), patch("quantopsy.io.os." + stage, side_effect=OSError("private setup details")):
                    with self.assertRaises((OSError, QuantopsyError)) as failure:
                        write_json_create(target, {"authored": True})
                with self.assertRaises(OSError):
                    os.fstat(descriptors[0])
                if stage == "fdopen":
                    self.assertFalse(target.exists())
                else:
                    # Without descriptor identity, preserving an uncertain path is
                    # safer than deleting a replacement. It is not a completed JSON record.
                    self.assertIsInstance(failure.exception, QuantopsyError)
                    self.assertEqual(failure.exception.code, "cleanup_incomplete")
                    self.assertTrue(target.exists())
                    self.assertEqual(target.read_bytes(), b"")

    def test_json_cleanup_preserves_a_replaced_path_after_the_owned_write_fails(self):
        output = self.directory / "replaced.json"
        retained = self.directory / "retained-owned.json"
        def replace_then_fail(_descriptor):
            output.rename(retained)
            output.write_bytes(b"independent replacement bytes")
            raise OSError("private flush failure")
        with patch("quantopsy.io.os.fsync", side_effect=replace_then_fail):
            with self.assertRaises((OSError, QuantopsyError)):
                write_json_create(output, {"authored": True})
        self.assertEqual(output.read_bytes(), b"independent replacement bytes")
        self.assertTrue(retained.is_file())
