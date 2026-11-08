"""Compare frozen, data-only task measurements without executing an engine.

Hashes identify the exact supplied records. Their contents remain caller claims;
this comparator cannot authenticate an external execution or a model's quality.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
from pathlib import Path
import re

from .contracts import DEFAULT_LIMITS
from .errors import fail
from .io import canonical_json_bytes, read_json_document
from .repair import MANIFEST_KIND, PLAN_KIND, validate_plan


SUITE_KIND = "quantopsy.behavior-suite"
CONFIG_KIND = "quantopsy.behavior-config"
MODELS_KIND = "quantopsy.behavior-models"
RUN_KIND = "quantopsy.behavior-run"
REPORT_KIND = "quantopsy.behavior-comparison"
MAX_TASKS = 128
MAX_METRICS = 8
_ROLES = ("reference", "quantized", "candidate")
_SHA256 = re.compile(r"[0-9a-f]{64}", re.ASCII)
_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", re.ASCII)
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]{0,17})(?:\.[0-9]{1,24})?", re.ASCII)
_THRESHOLDS = ("min_quantization_loss", "min_recovery_fraction",
               "max_reference_regression", "max_quantized_regression")


def _record(value: object, fields: set[str]) -> dict:
    if type(value) is not dict or set(value) != fields:
        fail("behavior_invalid_record", "Behavior records must contain exactly their supported fields.")
    return value


def _integer(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        fail("behavior_invalid_record", "Behavior counts must be bounded whole numbers.")
    return value


def _text(value: object, maximum: int, *, blank: bool = False) -> str:
    if type(value) is not str or (not blank and not value.strip()) or "\0" in value:
        fail("behavior_invalid_record", "Behavior text must be bounded ordinary UTF-8 text.")
    try:
        if len(value.encode("utf-8")) > maximum:
            fail("behavior_invalid_record", "Behavior text exceeds its UTF-8 byte limit.")
    except UnicodeError:
        fail("behavior_invalid_record", "Behavior text must contain valid Unicode.")
    return value


def _identifier(value: object) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        fail("behavior_invalid_record", "Behavior identifiers must be bounded letters, digits, dots, dashes or underscores.")
    return value


def _hash(value: object) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        fail("behavior_invalid_record", "Behavior hashes must be complete lowercase SHA-256 values.")
    return value


def _version(raw: dict, kind: str) -> None:
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1 or raw["kind"] != kind:
        fail("behavior_invalid_record", "Use the supported version-one behavior record kind.")


def _decimal(value: object, *, threshold: bool = False, fraction: bool = False) -> Decimal:
    if type(value) is not str or not _DECIMAL.fullmatch(value):
        fail("behavior_invalid_number", "Measurements and thresholds must be finite bounded decimal strings, without exponents.")
    number = Decimal(value)
    if (threshold and number < 0) or (fraction and not 0 <= number <= 1):
        fail("behavior_invalid_number", "Loss and regression limits must be nonnegative; recovery fractions must be between zero and one.")
    return number


def _number(value: Decimal) -> str:
    if value == 0:
        return "0"
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def _binding(value: object, *, candidate: bool = False) -> dict:
    raw = _record(value, {"name", "sha256", "size"})
    name = _text(raw["name"], 255)
    if name in (".", "..") or any(character in name for character in ("/", "\\", "\n", "\r")):
        fail("behavior_invalid_record", "Model labels must be filenames without directory paths.")
    limit = DEFAULT_LIMITS.max_output_bytes if candidate else DEFAULT_LIMITS.max_file_bytes
    return {"name": name, "sha256": _hash(raw["sha256"]), "size": _integer(raw["size"], 1, limit)}


def _engine(value: object) -> dict:
    raw = _record(value, {"name", "revision", "source_sha256"})
    return {"name": _identifier(raw["name"]), "revision": _text(raw["revision"], 256),
            "source_sha256": _hash(raw["source_sha256"])}


def validate_config(value: object) -> dict:
    raw = _record(value, {"schema_version", "kind", "engine", "tokenizer_sha256",
                          "context_tokens", "batch_tokens", "threads", "seed", "add_bos", "add_eos"})
    _version(raw, CONFIG_KIND)
    context = _integer(raw["context_tokens"], 2, 8192)
    for key in ("add_bos", "add_eos"):
        if type(raw[key]) is not bool:
            fail("behavior_invalid_record", "Tokenizer boundary flags must be booleans.")
    return {"schema_version": 1, "kind": CONFIG_KIND, "engine": _engine(raw["engine"]),
            "tokenizer_sha256": _hash(raw["tokenizer_sha256"]), "context_tokens": context,
            "batch_tokens": _integer(raw["batch_tokens"], 1, context),
            "threads": _integer(raw["threads"], 1, 64), "seed": _integer(raw["seed"], 0, 2**31 - 1),
            "add_bos": raw["add_bos"], "add_eos": raw["add_eos"]}


def validate_suite(value: object) -> dict:
    raw = _record(value, {"schema_version", "kind", "suite_id", "config_sha256", "metrics", "tasks"})
    _version(raw, SUITE_KIND)
    metrics = raw["metrics"]
    if type(metrics) is not list or not 1 <= len(metrics) <= MAX_METRICS:
        fail("behavior_invalid_suite", "Freeze between one and eight directed metrics before measuring.")
    seen = set()
    for metric in metrics:
        row = _record(metric, {"id", "direction", *_THRESHOLDS})
        name = _identifier(row["id"])
        if name in seen or type(row["direction"]) is not str or row["direction"] not in ("lower", "higher"):
            fail("behavior_invalid_suite", "Metrics must be unique and explicitly directed lower or higher.")
        seen.add(name)
        for key in _THRESHOLDS:
            _decimal(row[key], threshold=True, fraction=key == "min_recovery_fraction")
    tasks = raw["tasks"]
    if type(tasks) is not list or not 1 <= len(tasks) <= MAX_TASKS:
        fail("behavior_invalid_suite", "Freeze a nonempty suite of at most 128 tasks.")
    seen = set()
    for task in tasks:
        row = _record(task, {"id", "prompt", "target"})
        name = _identifier(row["id"])
        if name in seen:
            fail("behavior_invalid_suite", "Each frozen task must have a unique identifier.")
        seen.add(name)
        _text(row["prompt"], 16_384, blank=True)
        _text(row["target"], 16_384)
    return {"schema_version": 1, "kind": SUITE_KIND, "suite_id": _identifier(raw["suite_id"]),
            "config_sha256": _hash(raw["config_sha256"]), "metrics": metrics, "tasks": tasks}


def validate_models(value: object) -> dict:
    raw = _record(value, {"schema_version", "kind", *_ROLES})
    _version(raw, MODELS_KIND)
    return {"schema_version": 1, "kind": MODELS_KIND,
            **{role: _binding(raw[role], candidate=role == "candidate") for role in _ROLES}}


def validate_run(value: object, role: str, suite: dict, suite_sha256: str,
                 config: dict, config_sha256: str, model: dict) -> dict:
    raw = _record(value, {"schema_version", "kind", "role", "suite_sha256", "config_sha256",
                          "engine", "model", "status", "error", "tasks"})
    _version(raw, RUN_KIND)
    if role not in _ROLES or raw["role"] != role:
        fail("behavior_identity_mismatch", "Each measurement record must retain its explicitly supplied model role.")
    if (_hash(raw["suite_sha256"]) != suite_sha256 or _hash(raw["config_sha256"]) != config_sha256
            or _engine(raw["engine"]) != config or _binding(raw["model"], candidate=role == "candidate") != model):
        fail("behavior_identity_mismatch", "Measurements must bind the same exact suite, configuration, engine and role-specific model.")
    status = raw["status"]
    if type(status) is not str or status not in ("complete", "incomplete", "error"):
        fail("behavior_invalid_run", "Run status must be complete, incomplete or error.")
    if status == "complete":
        if raw["error"] is not None:
            fail("behavior_invalid_run", "A complete run cannot also declare an error.")
    else:
        _text(raw["error"], 2048)
    rows = raw["tasks"]
    if type(rows) is not list or len(rows) > len(suite["tasks"]) or (status == "complete" and len(rows) != len(suite["tasks"])):
        fail("behavior_coverage_mismatch", "Complete runs must cover every frozen task; failed runs may retain only an ordered completed prefix.")
    for expected, row in zip(suite["tasks"], rows):
        _record(row, {"id", "metrics"})
        if row["id"] != expected["id"]:
            fail("behavior_coverage_mismatch", "Run tasks must retain the exact frozen identifiers and order.")
        measured = row["metrics"]
        if type(measured) is not list or len(measured) != len(suite["metrics"]):
            fail("behavior_coverage_mismatch", "Each measured task must retain every frozen metric in order.")
        for metric, item in zip(suite["metrics"], measured):
            _record(item, {"id", "value"})
            if item["id"] != metric["id"]:
                fail("behavior_coverage_mismatch", "Task metric identifiers and order must match the frozen suite.")
            _decimal(item["value"])
    return raw


def _manifest(value: object, models: dict) -> None:
    raw = _record(value, {"schema_version", "kind", "plan_sha256", "reference", "quantized", "output",
                          "architecture", "selected", "added_weight_bytes", "max_added_bytes", "max_elements",
                          "reserve_bytes", "metadata_omissions", "provenance", "lineage", "behavioral_validation", "verification"})
    if raw["kind"] != MANIFEST_KIND or type(raw["schema_version"]) is not int or raw["schema_version"] not in (1, 2):
        fail("behavior_invalid_manifest", "Use a supported complete Quantopsy repair manifest.")
    _hash(raw["plan_sha256"])
    plan_fields = ("schema_version", "reference", "quantized", "architecture", "provenance", "selected",
                   "added_weight_bytes", "max_added_bytes", "max_elements", "lineage", "behavioral_validation")
    plan = validate_plan({"kind": PLAN_KIND, **{key: raw[key] for key in plan_fields}},
                         profile="v1" if raw["schema_version"] == 1 else "v2")
    if hashlib.sha256(canonical_json_bytes(plan)).hexdigest() != raw["plan_sha256"]:
        fail("behavior_invalid_manifest", "The repair manifest must bind its complete canonical repair plan.")
    _integer(raw["reserve_bytes"], 0, 2**53 - 1)
    if raw["metadata_omissions"] not in ([], ["general.file_type"]):
        fail("behavior_invalid_manifest", "The repair manifest has unsupported metadata omissions.")
    checks = _record(raw["verification"], {"container_reparsed", "metadata_preserved", "tensor_payloads_verified", "source_hashes_verified"})
    if any(type(item) is not bool or not item for item in checks.values()):
        fail("behavior_invalid_manifest", "The supplied repair manifest must declare all prescribed byte checks complete.")
    if (raw["reference"] != models["reference"] or raw["quantized"] != models["quantized"]
            or _binding(raw["output"], candidate=True) != models["candidate"]):
        fail("behavior_identity_mismatch", "The repair manifest must bind the complete supplied reference, quantized and candidate model identities.")


def _case(metric: dict, values: dict[str, str | None]) -> dict:
    result = {"metric": metric["id"], "direction": metric["direction"], "values": values,
              "thresholds": {key: metric[key] for key in _THRESHOLDS},
              "status": "unavailable", "quantization_loss": None, "qualifying_loss": False,
              "candidate_improvement": None, "recovery_fraction": None, "recovered": False,
              "reference_regression": None, "quantized_regression": None,
              "reference_regression_exceeded": False, "quantized_regression_exceeded": False}
    if any(value is None for value in values.values()):
        return result
    # Bounded input decimal lengths need fewer than 100 significant digits for
    # these differences and products. Threshold decisions never use division.
    with localcontext() as context:
        context.prec = 128
        direction = Decimal(1 if metric["direction"] == "lower" else -1)
        reference, quantized, candidate = (_decimal(values[role]) for role in _ROLES)
        loss = direction * (quantized - reference)
        improvement = direction * (quantized - candidate)
        reference_regression = direction * (candidate - reference)
        quantized_regression = -improvement
        qualifying = loss > 0 and loss >= _decimal(metric["min_quantization_loss"])
        recovered = qualifying and improvement > 0 and improvement >= loss * _decimal(metric["min_recovery_fraction"])
        result.update({"status": "compared", "quantization_loss": _number(loss), "qualifying_loss": qualifying,
                       "candidate_improvement": _number(improvement), "recovered": recovered,
                       "recovery_fraction": _number(improvement / loss) if loss > 0 else None,
                       "reference_regression": _number(reference_regression), "quantized_regression": _number(quantized_regression),
                       "reference_regression_exceeded": reference_regression > _decimal(metric["max_reference_regression"]),
                       "quantized_regression_exceeded": quantized_regression > _decimal(metric["max_quantized_regression"])})
    return result


def compare_behavior(suite_path: str | Path, config_path: str | Path, models_path: str | Path,
                     reference_run_path: str | Path, quantized_run_path: str | Path, candidate_run_path: str | Path,
                     *, manifest_path: str | Path | None = None) -> dict:
    """Check supplied measurements; perform no inference or model-file reads."""
    suite_value, suite_hash = read_json_document(suite_path)
    config_value, config_hash = read_json_document(config_path)
    models_value, models_hash = read_json_document(models_path)
    suite, config, models = validate_suite(suite_value), validate_config(config_value), validate_models(models_value)
    if suite["config_sha256"] != config_hash:
        fail("behavior_identity_mismatch", "The frozen suite must bind the exact configuration bytes used by all runs.")
    runs = {}
    hashes = {}
    for role, path in zip(_ROLES, (reference_run_path, quantized_run_path, candidate_run_path)):
        value, hashes[role] = read_json_document(path)
        runs[role] = validate_run(value, role, suite, suite_hash, config["engine"], config_hash, models[role])
    manifest_hash = None
    if manifest_path is not None:
        value, manifest_hash = read_json_document(manifest_path)
        _manifest(value, models)
    tasks = []
    for index, task in enumerate(suite["tasks"]):
        cases = []
        for metric_index, metric in enumerate(suite["metrics"]):
            values = {role: run["tasks"][index]["metrics"][metric_index]["value"] if index < len(run["tasks"]) else None
                      for role, run in runs.items()}
            cases.append(_case(metric, values))
        tasks.append({"id": task["id"], "cases": cases})
    all_cases = [case for task in tasks for case in task["cases"]]
    lost = sum(case["qualifying_loss"] for case in all_cases)
    recovered = sum(case["recovered"] for case in all_cases)
    regressions = sum(case["reference_regression_exceeded"] or case["quantized_regression_exceeded"] for case in all_cases)
    complete = all(run["status"] == "complete" for run in runs.values())
    measurements_pass = complete and lost > 0 and recovered == lost and regressions == 0
    result = {"schema_version": 1, "kind": REPORT_KIND, "suite_id": suite["suite_id"],
              "inputs": {"suite_sha256": suite_hash, "config_sha256": config_hash, "models_sha256": models_hash,
                         "run_sha256": hashes, "repair_manifest_sha256": manifest_hash},
              "engine": config["engine"], "models": {role: models[role] for role in _ROLES},
              "runs": {role: {"status": run["status"], "error": run["error"], "measured_tasks": len(run["tasks"])}
                       for role, run in runs.items()},
              "tasks": tasks,
              "acceptance": {"complete_runs": complete, "qualifying_loss_cases": lost, "recovered_cases": recovered,
                             "regression_cases": regressions, "measurements_pass": measurements_pass,
                             "repair_manifest_records_bound": manifest_hash is not None,
                             "supplied_recovery_records_pass": measurements_pass and manifest_hash is not None},
              "evidence": "caller_supplied_measurements_identity_checked",
              "inference_execution": "not_performed_by_comparator",
              "model_payloads": "not_opened_by_comparator",
              "freeze_chronology": "not_established_by_imported_records",
              "source_execution_authenticity": "not_established_by_imported_records",
              "behavioral_recovery": "requires_independently_verified_inference_evidence"}
    canonical_json_bytes(result)
    return result
