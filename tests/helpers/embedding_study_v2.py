"""Trusted CI-only embedding-study-v2 data and parser; no product execution.

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
    "tests/helpers/embedding_study_v2.py",
    "tests/helpers/behavioral_engine/CMakeLists.txt",
    "tests/scripts/run-embedding-study-v2.py",
    "docs/studies/embedding-v2.json",
)
PROTOCOL_PATH = "docs/studies/embedding-v2.json"
PROTOCOL_SHA256 = "8c35578fb0f8a2451a876f0a86cacf98941f7887a289df0bfd19c5ae416c5bc3"
PARAMETERS = {"context_tokens": 128, "batch_tokens": 128, "threads": 2,
              "seed": 0, "add_bos": True, "add_eos": False}
LEGACY_TENSORS = tuple(
    f"blk.{layer}.{matrix}.weight" for layer in range(6)
    for matrix in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
)
SELECTED_TENSORS = LEGACY_TENSORS + ("token_embd.weight",)
SELECTED_ELEMENTS = 15_187_968
ADDED_WEIGHT_BYTES = 52_208_640
METRICS = (
    {"id": "mean_target_nll", "direction": "lower", "min_quantization_loss": "0.005",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.02", "max_quantized_regression": "0.01"},
    {"id": "target_token_accuracy", "direction": "higher", "min_quantization_loss": "0.02",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.03", "max_quantized_regression": "0.02"},
)
# Independently authored for this one frozen study before any model execution.
# These are ordinary story continuations, not instructions to an assistant.
TASKS = (
    ("embedding-v2-01-scarf", "A cold wind blew as Max waited for the bus.\n", "His grandmother gave him a long scarf. He wrapped it around his neck and felt warm. When the bus came, he waved to her from the window. She waved back until the bus was out of sight."),
    ("embedding-v2-02-puzzle", "Ruby spread the puzzle pieces across the floor.\n", "She found the corners first and put them together. Her brother helped her look for the blue pieces. One piece was under his sock. When the picture was done, they could see a big ship on the sea."),
    ("embedding-v2-03-hat", "The puppy took Will's hat and ran into the garden.\n", "Will called the puppy in a kind voice. It stopped near the fence and looked at him. Will held out a toy, and the puppy dropped the hat. Then they played together on the soft grass."),
    ("embedding-v2-04-paint", "Ada wanted to paint a picture for her teacher.\n", "She chose a clean sheet of paper and filled a cup with water. First she painted a yellow sun. Then she added a house with a red door. She let the picture dry before putting it in her school bag."),
    ("embedding-v2-05-bridge", "Two children built a bridge with wooden blocks.\n", "The middle fell down when they tried to put a toy truck on it. They made the legs wider and tried again. This time the bridge stayed up. They laughed as the truck rolled safely to the other side."),
    ("embedding-v2-06-basket", "Owen helped his grandfather carry a heavy basket.\n", "They each held one handle and walked slowly along the path. At the kitchen door they put the basket down. Inside were carrots from the garden. Owen washed them while his grandfather got a pot for their soup."),
    ("embedding-v2-07-snow", "Fresh snow covered the hill behind the school.\n", "Ivy pulled her sled to the top and sat down. Her friend gave it a small push. The sled went quickly down the hill and stopped near a tree. Ivy got up, smiled, and pulled it back for her friend."),
    ("embedding-v2-08-bell", "A small bell rang when June opened the shop door.\n", "The woman behind the counter smiled and said hello. June showed her the list from her father. The woman helped her find bread and milk. June paid, thanked her, and carried the bag home with both hands."),
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


def protocol_contract(repo: str | Path) -> dict:
    raw = (Path(repo) / PROTOCOL_PATH).read_bytes()
    need(sha(raw) == PROTOCOL_SHA256, "predeclared_protocol_bytes_changed")
    value = strict_json(raw)
    need(value["kind"] == "quantopsy.predeclared-embedding-study" and value["schema_version"] == 2
         and value["study_id"] == "tinystories-input-embedding-v2"
         and value["tasks"] == [{"id": identity, "prompt": prompt, "target": target}
                                for identity, prompt, target in TASKS]
         and value["metrics"] == [dict(metric) for metric in METRICS], "predeclared_tasks_or_metrics_changed")
    engine = value["engine"]
    need(engine["revision"] == UPSTREAM_REVISION and engine["git_tree"] == UPSTREAM_TREE
         and engine["parameters"] == {**PARAMETERS, "sampling": False, "cpu_only": True}, "predeclared_engine_changed")
    strategy = value["candidate_strategy"]
    legacy, embedding = strategy["legacy_internal_matrices"], strategy["embedding_extension"]
    need(legacy["layers"] == list(range(6))
         and legacy["matrix_names_in_order"] == ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
         and legacy["profile"] == "v2" and legacy["count"] == 42
         and legacy["selected_elements"] == 5_971_968 and legacy["added_weight_bytes"] == 20_528_640
         and embedding["profile"] == "v3" and embedding["append_tensor"] == "token_embd.weight"
         and embedding["count"] == 43 and embedding["selected_elements"] == SELECTED_ELEMENTS
         and embedding["added_weight_bytes"] == ADDED_WEIGHT_BYTES
         and strategy["automatic_selection"] is False, "predeclared_candidate_strategy_changed")
    return value


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
    return {"schema_version": 1, "kind": "quantopsy.behavior-suite", "suite_id": "tinystories-input-embedding-v2",
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
