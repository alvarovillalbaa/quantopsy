"""Trusted CI-only frozen task data and parser; no application imports or execution.

The owning controller builds/invokes the engine, retains its original streams,
binds files and enforces resource/process/teardown limits. These helpers perform
no download, subprocess invocation, model execution or filesystem mutation.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
from pathlib import Path
import re


UPSTREAM_REVISION = "74d4f5b041ad837153b0e90fc864b8290e01d8d5"
UPSTREAM_TREE = "9ae6f9dd0984b780c942ccf510e9cd5fcee9e724"
UPSTREAM_GITLINKS = (
    {"path": "ggml/src/ggml-kompute/kompute", "mode": "160000",
     "sha1": "4565194ed7c32d1d2efa32ceab4d3c6cae006306"},
)
ENGINE_REVISION = UPSTREAM_REVISION + "+quantopsy-target-nll-v1"
SOURCE_PATHS = (
    "tests/helpers/behavioral_scorer.cpp",
    "tests/helpers/behavioral_inference.py",
    "tests/helpers/behavioral_engine/CMakeLists.txt",
)
PARAMETERS = {"context_tokens": 128, "batch_tokens": 128, "threads": 2,
              "seed": 0, "add_bos": True, "add_eos": False}
SELECTED_TENSORS = tuple(
    f"blk.{layer}.{matrix}.weight" for layer in range(6)
    for matrix in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
)
SELECTED_ELEMENTS = 5_971_968
ADDED_WEIGHT_BYTES = 20_528_640
METRICS = (
    {"id": "mean_target_nll", "direction": "lower", "min_quantization_loss": "0.005",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.02", "max_quantized_regression": "0.01"},
    {"id": "target_token_accuracy", "direction": "higher", "min_quantization_loss": "0.02",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.03", "max_quantized_regression": "0.02"},
)
# Independently authored for this one frozen study before any model execution.
# These are ordinary story continuations, not instructions to an assistant.
TASKS = (
    ("story-01-boat", "Mia put her little boat in the pond.\n",
     "A soft wind pushed it away. Mia ran to the other side and waited. Soon the boat came back to her. "
     "She picked it up and smiled. Then she took it home so it would be safe."),
    ("story-02-seed", "Tom found a small seed near the gate.\n",
     "He put the seed in a pot and gave it water. Each day he looked at the pot. At first he saw only dirt. "
     "Then a little green leaf came up. Tom was glad that he had waited."),
    ("story-03-key", "Ben could not find the key to his box.\n",
     "He looked under the bed and behind the door. The key was not there. His sister saw it beside his shoe. "
     "She gave it to him. Ben opened the box and took out a toy car to share with her."),
    ("story-04-rain", "Lily saw dark clouds above the park.\n",
     "She put her ball in her bag and walked home. Just as she shut the door, rain began to fall. "
     "Lily was dry and warm inside. She sat by the window and watched the drops run down the glass."),
    ("story-05-sharing", "Sam had two apples and his friend had none.\n",
     "Sam gave one apple to his friend. They sat on the grass and ate together. His friend said thank you. "
     "After lunch they played a game with a red ball. Sam was happy because they could both have a good day."),
    ("story-06-bird", "A little bird sat on a low branch.\n",
     "Nina stopped to look at it. She stayed very still and did not make a sound. The bird sang a short song. "
     "Then it flew up to a tall tree. Nina smiled and told her mother about the song when she got home."),
    ("story-07-cake", "Dad put a small cake on the table.\n",
     "It was still too hot to eat. Jack wanted a piece, but he waited while Dad made some tea. "
     "Soon the cake was cool. Dad cut it into pieces and gave one to Jack. They ate the cake together and talked."),
    ("story-08-light", "Eva went into her room after the sun went down.\n",
     "The room was dark, so she turned on the light. Now she could see her book on the chair. "
     "She picked it up and sat on the bed. Her cat jumped up beside her, and Eva read a story to the cat."),
)
_SHA = re.compile(r"[0-9a-f]{64}", re.ASCII)
_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", re.ASCII)
_RAW_NLL = re.compile(r"(?:0|[1-9][0-9]{0,3})\.[0-9]{17}", re.ASCII)
_ROLES = ("reference", "quantized", "candidate")


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("ascii")


def strict_json(raw: bytes, limit: int = 262144) -> object:
    need(type(raw) is bytes and 0 < len(raw) <= limit, "json_original_bound")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "json_duplicate_key")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("json_nonfinite")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)


def engine_manifest(repo: str | Path) -> dict:
    """Bind scoring, aggregation and build recipe together without circular hashes."""
    root = Path(repo)
    sources = []
    for relative in SOURCE_PATHS:
        path = root / relative
        need(path.is_file() and not path.is_symlink(), "engine_source_regular_file_required")
        raw = path.read_bytes()
        need(0 < len(raw) <= 131072, "engine_source_bound")
        sources.append({"path": relative, "bytes": len(raw), "sha256": sha(raw)})
    return {
        "schema_version": 1, "kind": "quantopsy.trusted-cpu-engine-source",
        "upstream": {"repository": "ggml-org/llama.cpp", "revision": UPSTREAM_REVISION,
                     "git_tree_sha1": UPSTREAM_TREE, "release": "b5046",
                     "omitted_archive_gitlinks": [dict(row) for row in UPSTREAM_GITLINKS]},
        "sources": sources,
        "recipe": {
            "target": "quantopsy-behavioral-scorer", "parallel_build_jobs": 2,
            "parameters": dict(PARAMETERS), "kv_cache": "F32", "cpu_only": True,
            "native_cpu_tuning": False, "openmp": False, "flash_attention": False,
            "curl": False, "dynamic_backends": False, "sampling": False,
            "seed_usage": "recorded_unused_no_sampling", "context_overflow": "refuse_no_truncation",
            "token_boundary": "joint_tokenize_exact_prompt_prefix", "parse_special": False,
            "nll": "stable_log_softmax_serial_double", "accuracy_ties": "lowest_token_id",
            "aggregation": "Decimal_mean_fixed_12_places_ROUND_HALF_EVEN",
        },
    }


def tokenizer_fingerprint(metadata: list[dict]) -> str:
    """Hash exact encoded tokenizer metadata signatures sorted by metadata key."""
    need(type(metadata) is list, "tokenizer_metadata_list_required")
    fields = ("key", "kind", "arrayKind", "count", "encodedBytes", "encodedSha256")
    selected = sorted(({key: row[key] for key in fields} for row in metadata if row["key"].startswith("tokenizer.")),
                      key=lambda row: row["key"])
    need(selected and len(selected) <= 128 and len({row["key"] for row in selected}) == len(selected), "tokenizer_metadata_coverage")
    for row in selected:
        need(type(row["encodedSha256"]) is str and _SHA.fullmatch(row["encodedSha256"]), "tokenizer_metadata_hash_required")
    return sha(canonical(selected))


def engine_config(engine_manifest_raw: bytes, tokenizer_sha256: str) -> dict:
    manifest = strict_json(engine_manifest_raw)
    need(manifest["kind"] == "quantopsy.trusted-cpu-engine-source"
         and manifest["upstream"]["revision"] == UPSTREAM_REVISION
         and manifest["upstream"]["git_tree_sha1"] == UPSTREAM_TREE
         and manifest["upstream"]["omitted_archive_gitlinks"] == list(UPSTREAM_GITLINKS), "engine_source_manifest_identity")
    need(type(tokenizer_sha256) is str and _SHA.fullmatch(tokenizer_sha256), "tokenizer_hash_required")
    return {"schema_version": 1, "kind": "quantopsy.behavior-config",
            "engine": {"name": "quantopsy-target-nll", "revision": ENGINE_REVISION,
                       "source_sha256": sha(engine_manifest_raw)},
            "tokenizer_sha256": tokenizer_sha256, **PARAMETERS}


def frozen_suite(config_raw: bytes) -> dict:
    config = strict_json(config_raw)
    need(config["engine"]["revision"] == ENGINE_REVISION
         and all(type(config[key]) is type(value) and config[key] == value for key, value in PARAMETERS.items()),
         "frozen_configuration_mismatch")
    return {"schema_version": 1, "kind": "quantopsy.behavior-suite", "suite_id": "tinystories-target-scores-v1",
            "config_sha256": sha(config_raw), "metrics": [dict(metric) for metric in METRICS],
            "tasks": [{"id": identity, "prompt": prompt, "target": target} for identity, prompt, target in TASKS]}


def encode_tasks(suite: dict) -> bytes:
    need(suite["tasks"] == [{"id": identity, "prompt": prompt, "target": target} for identity, prompt, target in TASKS],
         "authored_frozen_tasks_changed")
    rows = []
    for task in suite["tasks"]:
        need(_IDENTIFIER.fullmatch(task["id"]), "task_identifier_invalid")
        rows.append("\t".join((task["id"], task["prompt"].encode("utf-8").hex(), task["target"].encode("utf-8").hex())))
    raw = ("\n".join(rows) + "\n").encode("ascii")
    need(len(raw) <= 32768, "task_wire_bound")
    return raw


def _record(value: object, fields: set[str], reason: str) -> dict:
    need(type(value) is dict and set(value) == fields, reason)
    return value


def collect_run(raw_stdout: bytes, role: str, suite_raw: bytes, config_raw: bytes, model: dict) -> dict:
    """Check exact scorer coverage and independently aggregate its token originals."""
    need(role in _ROLES, "run_role_required")
    _record(model, {"name", "sha256", "size"}, "model_binding_fields")
    need(type(model["name"]) is str and model["name"] and "/" not in model["name"] and "\\" not in model["name"]
         and type(model["sha256"]) is str and _SHA.fullmatch(model["sha256"])
         and type(model["size"]) is int and 0 < model["size"] <= 16 * 1024**3, "model_binding_invalid")
    suite, config = strict_json(suite_raw), strict_json(config_raw)
    need(suite == frozen_suite(config_raw) and suite["config_sha256"] == sha(config_raw), "suite_freeze_mismatch")
    raw = _record(strict_json(raw_stdout), {"schema_version", "kind", "engine_revision", "parameters",
                                          "model_context_tokens", "vocab_tokens", "tasks"}, "scorer_root_fields")
    need(type(raw["schema_version"]) is int and raw["schema_version"] == 1
         and raw["kind"] == "quantopsy.llama-target-scores" and raw["engine_revision"] == ENGINE_REVISION,
         "scorer_identity_mismatch")
    parameters = _record(raw["parameters"], set(PARAMETERS), "scorer_parameter_fields")
    need(all(type(parameters[key]) is type(value) and parameters[key] == value for key, value in PARAMETERS.items()),
         "scorer_parameters_mismatch")
    need(type(raw["model_context_tokens"]) is int and 128 <= raw["model_context_tokens"] <= 8192
         and type(raw["vocab_tokens"]) is int and 1 <= raw["vocab_tokens"] <= 65536, "scorer_model_limits")
    need(type(raw["tasks"]) is list and len(raw["tasks"]) == len(TASKS), "scorer_task_coverage")
    run_rows, detail_rows = [], []
    for expected, current in zip(suite["tasks"], raw["tasks"]):
        row = _record(current, {"id", "prompt_tokens", "token_ids", "target_ids", "predicted_ids", "target_nll"}, "scorer_task_fields")
        need(row["id"] == expected["id"] and type(row["prompt_tokens"]) is int and 1 <= row["prompt_tokens"] < 128,
             "scorer_task_identity_or_prefix")
        ids, targets, predicted, values = (row[key] for key in ("token_ids", "target_ids", "predicted_ids", "target_nll"))
        need(all(type(sequence) is list for sequence in (ids, targets, predicted, values))
             and row["prompt_tokens"] < len(ids) <= 128 and 1 <= len(targets) <= 127
             and len(targets) == len(predicted) == len(values) and ids[row["prompt_tokens"]:] == targets,
             "scorer_token_coverage")
        need(all(type(token) is int and 0 <= token < raw["vocab_tokens"] for token in ids + targets + predicted),
             "scorer_token_ids_invalid")
        need(all(type(value) is str and _RAW_NLL.fullmatch(value) for value in values), "scorer_nll_encoding")
        with localcontext() as context:
            context.prec = 128
            numbers = [Decimal(value) for value in values]
            need(all(Decimal(0) <= value <= Decimal(1000) for value in numbers), "scorer_nll_range")
            total = sum(numbers, Decimal(0))
            correct = sum(target == actual for target, actual in zip(targets, predicted))
            quantum = Decimal("0.000000000001")
            mean_nll = format((total / len(numbers)).quantize(quantum, rounding=ROUND_HALF_EVEN), "f")
            accuracy = format((Decimal(correct) / len(numbers)).quantize(quantum, rounding=ROUND_HALF_EVEN), "f")
        run_rows.append({"id": row["id"], "metrics": [{"id": "mean_target_nll", "value": mean_nll},
                                                     {"id": "target_token_accuracy", "value": accuracy}]})
        detail_rows.append({"id": row["id"], "prompt_tokens": row["prompt_tokens"], "target_tokens": len(targets),
                            "correct_target_tokens": correct, "token_ids_sha256": sha(canonical(ids)),
                            "target_ids_sha256": sha(canonical(targets)), "predicted_ids_sha256": sha(canonical(predicted)),
                            "target_nll_sha256": sha(canonical(values)), "total_target_nll": format(total, "f"),
                            "mean_target_nll": mean_nll, "target_token_accuracy": accuracy})
    run = {"schema_version": 1, "kind": "quantopsy.behavior-run", "role": role,
           "suite_sha256": sha(suite_raw), "config_sha256": sha(config_raw), "engine": config["engine"],
           "model": dict(model), "status": "complete", "error": None, "tasks": run_rows}
    return {"run": run, "details": {"scorer_stdout_sha256": sha(raw_stdout), "scorer_stdout_bytes": len(raw_stdout),
                                    "model_context_tokens": raw["model_context_tokens"], "vocab_tokens": raw["vocab_tokens"],
                                    "tasks": detail_rows}}


def compare_tokenization(details_by_role: dict) -> None:
    """Comparability guard only; token identity is never a quality metric."""
    need(type(details_by_role) is dict and set(details_by_role) == set(_ROLES), "token_comparison_roles")
    fields = ("id", "prompt_tokens", "target_tokens", "token_ids_sha256", "target_ids_sha256")
    signature = lambda value: [{key: row[key] for key in fields} for row in value["tasks"]]
    reference = signature(details_by_role["reference"])
    need(all(signature(details_by_role[role]) == reference for role in _ROLES), "model_tokenization_differs")
    need(all(details_by_role[role]["vocab_tokens"] == details_by_role["reference"]["vocab_tokens"] for role in _ROLES),
         "model_vocabulary_size_differs")
