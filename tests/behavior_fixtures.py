"""Independently authored JSON records; identities are invented, never weights.

These records assert arithmetic expectations only. No engine was executed and
no supplied source, model, measurement or repair claim is authenticated here.
"""

from copy import deepcopy
import hashlib
import json


ROLES = ("reference", "quantized", "candidate")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bind_manifest_plan(documents):
    """Hash the independently assembled plan copied into this authored manifest."""
    manifest = documents["manifest"]
    fields = ("schema_version", "reference", "quantized", "architecture", "provenance", "selected",
              "added_weight_bytes", "max_added_bytes", "max_elements", "lineage", "behavioral_validation")
    plan = {"kind": "quantopsy.repair-plan", **{key: manifest[key] for key in fields}}
    manifest["plan_sha256"] = digest(encoded(plan))
    return documents


def bind_records(documents):
    """Rebind edited fixture bytes without changing model or engine identities."""
    config_hash = digest(encoded(documents["config"]))
    documents["suite"]["config_sha256"] = config_hash
    suite_hash = digest(encoded(documents["suite"]))
    for role in ROLES:
        documents[role]["config_sha256"] = config_hash
        documents[role]["suite_sha256"] = suite_hash
    return documents


def authored_documents():
    engine = {"name": "authored-engine", "revision": "fixture-r1",
              "source_sha256": "1" * 64}
    config = {"schema_version": 1, "kind": "quantopsy.behavior-config",
              "engine": engine, "tokenizer_sha256": "2" * 64,
              "context_tokens": 256, "batch_tokens": 64, "threads": 1,
              "seed": 17, "add_bos": True, "add_eos": False}
    models = {"schema_version": 1, "kind": "quantopsy.behavior-models",
              "reference": {"name": "invented-reference.gguf", "sha256": "3" * 64, "size": 1024},
              "quantized": {"name": "invented-quantized.gguf", "sha256": "4" * 64, "size": 768},
              "candidate": {"name": "invented-candidate.gguf", "sha256": "5" * 64, "size": 960}}
    suite = {"schema_version": 1, "kind": "quantopsy.behavior-suite",
             "suite_id": "authored-arithmetic", "config_sha256": "0" * 64,
             "metrics": [
                 {"id": "loss", "direction": "lower", "min_quantization_loss": "1",
                  "min_recovery_fraction": "0.5", "max_reference_regression": "1",
                  "max_quantized_regression": "0"},
                 {"id": "accuracy", "direction": "higher", "min_quantization_loss": "0.1",
                  "min_recovery_fraction": "0.5", "max_reference_regression": "0.2",
                  "max_quantized_regression": "0"}],
             "tasks": [{"id": "task-a", "prompt": "Authored prompt A", "target": "A"},
                       {"id": "task-b", "prompt": "Authored prompt B", "target": "B"}]}
    documents = {"config": config, "suite": suite, "models": models}
    values = {"reference": (("1", "0.9"), ("2", "0.8")),
              "quantized": (("3", "0.5"), ("4", "0.4")),
              "candidate": (("2", "0.7"), ("3", "0.6"))}
    for role in ROLES:
        documents[role] = {"schema_version": 1, "kind": "quantopsy.behavior-run",
                           "role": role, "suite_sha256": "0" * 64, "config_sha256": "0" * 64,
                           "engine": deepcopy(engine), "model": deepcopy(models[role]),
                           "status": "complete", "error": None,
                           "tasks": [{"id": task["id"], "metrics": [
                               {"id": "loss", "value": scores[0]},
                               {"id": "accuracy", "value": scores[1]}]}
                               for task, scores in zip(suite["tasks"], values[role])]}
    documents["manifest"] = {
        "schema_version": 1, "kind": "quantopsy.repair-result", "plan_sha256": "6" * 64,
        "reference": deepcopy(models["reference"]), "quantized": deepcopy(models["quantized"]),
        "output": deepcopy(models["candidate"]), "architecture": "llama",
        "selected": [{"name": "blk.0.attn_q.weight", "dimensions": [32, 2], "elements": 64,
                      "reference_kind": 1, "quantized_kind": 2,
                      "reference_sha256": "7" * 64, "quantized_sha256": "8" * 64,
                      "reference_bytes": 128, "quantized_bytes": 36}],
        "added_weight_bytes": 92, "max_added_bytes": 92, "max_elements": 64,
        "reserve_bytes": 0, "metadata_omissions": ["general.file_type"],
        "provenance": {"model_id": "authored/invented", "revision": "fixture-r1",
                       "license": "independently authored fixture only",
                       "reference_source": "authored identity; no model file",
                       "quantized_source": "authored identity; no model file"},
        "lineage": "operator_declared_compatibility_checked", "behavioral_validation": "not_run",
        "verification": {"container_reparsed": True, "metadata_preserved": True,
                         "tensor_payloads_verified": True, "source_hashes_verified": True}}
    return bind_records(bind_manifest_plan(documents))
