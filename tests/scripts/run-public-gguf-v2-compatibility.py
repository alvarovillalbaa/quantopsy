#!/usr/bin/env python3
"""Owned opt-in v2 fixture/real-public-file compatibility and candidate checks."""
import base64
import collections
import datetime
import gzip
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import secrets
import selectors
import shutil
import signal
import socket
import ssl
import stat
import struct
import zlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_COMMIT = "3f3b3aedabb85e5909db6910df6c67a2d3873b18"
BASE_BYTES = 811425
BASE_STUB_BLOB = "c3bc1790fd252068730a73e607d19f9c99a0a451"
BASE_STUB_SHA256 = "934fa2a1c33bb308d30232bb6e5795f6e7caced206cd7a0f071591fba0f769fc"
BASE_STUB_BYTES = 408
REPOSITORY = "alvarovillalbaa/quantopsy"
BRANCH = "refs/heads/qa/quantopsy-compat-v2-20261107-1"
ACTIVE = ".github/workflows/gguf-compat-v2.yml"
CONTROLLER = "tests/scripts/run-public-gguf-v2-compatibility.py"
HELPER = "tests/helpers/public_gguf_v2_bindings.py"
HELPER_SHA256 = "0b90498f1386408f7936c5d685a4a8a1fc41b0a99eaac1f56db2a6f851b4aafc"
INVENTORY_HELPER = "tests/helpers/public_gguf_inventory.py"
INVENTORY_SHA256 = "e1a51e561fc9ba2ab0a2b4e3604575accd37d2b560eebd7cda62772bda32c1f5"
MODIFIED = {
    "quantopsy/contracts.py": {
        "bytes": 1400,
        "sha256": "b9ec1b2c9a334012fd31310bc138c2bef63a7258ca67cf0c5d53157f9af414a1"
    },
    "quantopsy/gguf.py": {
        "bytes": 20360,
        "sha256": "d62457ce52b03f8e5467282c598766fc4bec93162e4df14deb93b752bc5f5107"
    },
    "quantopsy/repair.py": {
        "bytes": 31337,
        "sha256": "8cb54bead8f17a8c17ac3e7929eab09ac414b51fbfbd97c14891364c80971c3f"
    },
    "quantopsy/cli.py": {
        "bytes": 7218,
        "sha256": "0e3d692a80d22f44030c204ff0b71c506278b4464be84ece6316cea442e2d1b0"
    },
    "AGENTS.md": {
        "bytes": 7666,
        "sha256": "1ebf6075a041ec2fce15e79e16d53a06f735301293395d5e52f05978ed325b3a"
    },
    "README.md": {
        "bytes": 6125,
        "sha256": "c19ae1f3bb09ff8885a986c2bde2f1bebf841fc81c8049c4e578f308083d8ff8"
    },
    "docs/format.md": {
        "bytes": 8637,
        "sha256": "4389204ed7eb829e48704b6f98ce2b5355af2de15e31f234a11df97d52457b34"
    },
    "docs/sources.md": {
        "bytes": 3321,
        "sha256": "929c56660d254a8b98c357c338ea0d3e8f7cee2db49baf781effd4dcf4187554"
    },
    "tests/README.md": {
        "bytes": 4333,
        "sha256": "9b885d2941488219d5dd47b3487b9b1c554e048e32962128f92bb946f0adf0be"
    }
}
ADDED = sorted([HELPER, "tests/helpers/v2_fixtures.py", "tests/helpers/__init__.py",
                "tests/integration/__init__.py", "tests/integration/test_profile_v2.py", CONTROLLER,
                "docs/proposals/opaque-q8-f32-v2.md", "docs/proposals/gguf-compat-v2.workflow.yml.example"])
NEW_PINS = {
    HELPER: {"bytes": 28139, "sha256": HELPER_SHA256},
    "tests/integration/test_profile_v2.py": {"bytes": 31722, "sha256": "2fe837dd6866ff469e5754a3f889cde25f0ecc81fa9a9e410ee70cb57d4d13a2"},
    "tests/helpers/v2_fixtures.py": {"bytes": 2981, "sha256": "d20977ba1f125b6221c9dce90abe81bad136b76ab433d43b9470aa2c49a230a6"},
}
FIXTURE_MODULES = ("tests.test_gguf", "tests.test_repair", "tests.test_cli", "tests.integration.test_profile_v2")
FIXTURE_IDS = [
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_both_readers_and_operation_must_agree_on_the_explicit_profile",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_changed_source_before_or_during_v2_copy_never_publishes_completion",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_cli_plan_and_repair_require_explicit_matching_version_and_publish_schema_two",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_cli_v2_inspection_is_opt_in_and_diagnostics_reject_profile_flags",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_default_v1_refuses_q8_and_keeps_explicit_v1_reports_and_candidate_bytes",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_dense_layout_fences_remain_required_under_v2",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_every_unquantized_f32_and_f16_type_and_payload_binding_is_required",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_failed_v2_install_preserves_replaced_inodes_and_unexpected_files",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_forged_source_and_selected_hashes_cannot_write_a_candidate",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_invalid_profile_values_are_not_coerced_and_fail_before_input_open",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_manifest_install_failure_removes_owned_v2_artifacts_without_completion",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_opaque_q8_decoder_refuses_before_any_tensor_stream_is_consumed",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_pairing_checks_encoded_metadata_by_key_without_requiring_the_same_order",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_pairing_rejects_changed_complete_tensor_names_or_dimensions",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_q8_requires_whole_dimension_zero_rows_not_just_total_blocks",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_q8_span_is_two_explicit_34_byte_blocks_and_profile_is_read_only",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_q8_truncation_overlap_alignment_and_unknown_encodings_are_refused",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_q8_without_q4_still_requires_exact_quantization_version_two",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_reference_inventory_cannot_contain_opaque_q8",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_schema_versions_cannot_silently_choose_or_upgrade_the_requested_profile",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_selected_f32_values_and_q4_scales_must_be_finite",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_v2_create_only_outputs_preserve_existing_files_directories_and_symlinks",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_v2_output_and_storage_budgets_fail_before_directory_creation",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_v2_plan_binds_selected_f32_bytes_and_returns_detached_strict_records",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_v2_plan_rejects_boolean_integer_coercions_extra_fields_and_f16_widths",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_v2_selection_refuses_q8_quantized_tensors_and_f16_references",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_v2_writer_reparses_both_stages_and_preserves_every_prescribed_byte",
    "tests.integration.test_profile_v2.ProfileV2IntegrationTier2Tests.test_work_and_growth_budgets_are_checked_before_selected_value_decoding",
    "tests.test_cli.CliTests.test_cli_failures_return_finite_json_without_echoing_paths_or_argument_payloads",
    "tests.test_cli.CliTests.test_cli_report_output_refuses_existing_files_and_retains_their_exact_bytes",
    "tests.test_cli.CliTests.test_diagnose_command_uses_selected_actual_weights_without_a_runner",
    "tests.test_cli.CliTests.test_failed_json_fsync_removes_only_the_owned_partial_and_closes_its_descriptor",
    "tests.test_cli.CliTests.test_inspect_prints_actual_container_hash_and_explicit_unrun_behavior",
    "tests.test_cli.CliTests.test_json_cleanup_preserves_a_replaced_path_after_the_owned_write_fails",
    "tests.test_cli.CliTests.test_json_create_is_private_and_input_output_paths_refuse_symlinks_or_nonfiles",
    "tests.test_cli.CliTests.test_json_depth_entry_byte_and_limit_budgets_reject_before_unbounded_output",
    "tests.test_cli.CliTests.test_json_descriptor_setup_failures_close_the_fd_and_never_claim_a_completed_record",
    "tests.test_cli.CliTests.test_json_output_rejects_cycles_non_json_values_and_isolated_unicode_units",
    "tests.test_cli.CliTests.test_json_reads_reject_invalid_utf8_duplicates_surrogates_and_nonfinite_numbers",
    "tests.test_cli.CliTests.test_json_roundtrip_preserves_unicode_zero_and_canonical_key_order",
    "tests.test_cli.CliTests.test_pair_open_failure_closes_prior_resources_through_the_actual_cli_stack",
    "tests.test_cli.CliTests.test_plan_command_creates_a_new_provenance_bound_record_with_exact_source_hashes",
    "tests.test_cli.CliTests.test_probe_command_reads_explicit_calibration_and_reports_only_supplied_linear_evidence",
    "tests.test_cli.CliTests.test_repair_command_writes_real_selected_f16_bytes_without_subprocess_or_network",
    "tests.test_gguf.GGUFTests.test_alignment_accepts_multiples_of_eight_and_rejects_invalid_metadata",
    "tests.test_gguf.GGUFTests.test_declared_metadata_and_tensor_counts_are_bounded_before_allocation",
    "tests.test_gguf.GGUFTests.test_duplicate_metadata_and_tensor_names_are_refused",
    "tests.test_gguf.GGUFTests.test_f16_values_preserve_independently_packed_binary16_edges",
    "tests.test_gguf.GGUFTests.test_f32_values_preserve_sign_fraction_and_negative_zero",
    "tests.test_gguf.GGUFTests.test_fifo_refusal_and_nonblocking_nofollow_descriptor_flags_are_explicit",
    "tests.test_gguf.GGUFTests.test_file_string_metadata_array_and_global_value_budgets_are_explicit",
    "tests.test_gguf.GGUFTests.test_invalid_utf8_in_keys_values_and_tensor_names_is_not_replaced",
    "tests.test_gguf.GGUFTests.test_magic_endianness_and_unsupported_versions_are_not_guessed",
    "tests.test_gguf.GGUFTests.test_nonfinite_f32_f16_and_q4_scale_values_fail_during_decode",
    "tests.test_gguf.GGUFTests.test_one_byte_streams_and_hashes_cover_exact_payloads_and_fresh_file",
    "tests.test_gguf.GGUFTests.test_overlapping_payload_spans_are_rejected_independent_of_table_order",
    "tests.test_gguf.GGUFTests.test_payload_truncation_cannot_be_hidden_by_valid_header_and_offsets",
    "tests.test_gguf.GGUFTests.test_q4_low_nibbles_precede_high_nibbles_for_one_literal_block",
    "tests.test_gguf.GGUFTests.test_q4_negative_and_zero_scales_preserve_rows_without_positive_fabrication",
    "tests.test_gguf.GGUFTests.test_q4_rows_and_quantization_version_must_match_the_supported_profile",
    "tests.test_gguf.GGUFTests.test_scalar_and_flat_array_metadata_have_independent_encodings",
    "tests.test_gguf.GGUFTests.test_shard_metadata_and_tokenizer_only_zero_tensor_files_are_rejected",
    "tests.test_gguf.GGUFTests.test_source_mutation_and_path_replacement_invalidate_open_streams_and_hashes",
    "tests.test_gguf.GGUFTests.test_symlink_leaf_and_directories_are_never_read_as_model_inputs",
    "tests.test_gguf.GGUFTests.test_tensor_relative_offsets_must_be_aligned",
    "tests.test_gguf.GGUFTests.test_truncated_headers_and_declared_strings_fail_without_large_buffers",
    "tests.test_gguf.GGUFTests.test_unknown_nested_boolean_and_nonfinite_metadata_types_are_rejected",
    "tests.test_gguf.GGUFTests.test_unsupported_tensor_encodings_do_not_receive_guessed_sizes",
    "tests.test_gguf.GGUFTests.test_v3_inventory_identity_and_original_metadata_bytes",
    "tests.test_gguf.GGUFTests.test_zero_excessive_rank_and_overflowing_dimensions_are_rejected",
    "tests.test_repair.RepairTests.test_added_weight_and_decode_work_budgets_accept_exact_bounds_before_decode",
    "tests.test_repair.RepairTests.test_all_selected_matrices_use_reference_encoding_without_reordering_inventory",
    "tests.test_repair.RepairTests.test_candidate_restores_only_selected_reference_bytes_and_preserves_metadata",
    "tests.test_repair.RepairTests.test_dense_llama_profile_rejects_other_architectures_experts_and_fused_layouts",
    "tests.test_repair.RepairTests.test_existing_directory_file_and_symlink_outputs_are_never_overwritten",
    "tests.test_repair.RepairTests.test_explicit_plan_binds_selected_bytes_shapes_input_hashes_and_provenance",
    "tests.test_repair.RepairTests.test_input_mutation_before_or_during_copy_never_publishes_completion",
    "tests.test_repair.RepairTests.test_interruption_cleanup_removes_only_owned_inodes_and_keeps_unexpected_files",
    "tests.test_repair.RepairTests.test_manifest_is_published_last_after_reparse_and_contains_exact_artifact_hashes",
    "tests.test_repair.RepairTests.test_non_power_of_two_alignment_is_preserved_with_rebuilt_offsets",
    "tests.test_repair.RepairTests.test_output_size_storage_and_reserve_budgets_fail_before_bundle_creation",
    "tests.test_repair.RepairTests.test_pair_rejects_architecture_and_nonquantization_identity_disagreement",
    "tests.test_repair.RepairTests.test_pair_requires_complete_tensor_names_and_exact_dimensions",
    "tests.test_repair.RepairTests.test_pair_summary_binds_complete_original_bytes_and_honest_lineage",
    "tests.test_repair.RepairTests.test_plan_validation_is_detached_strict_and_checks_internal_arithmetic",
    "tests.test_repair.RepairTests.test_provenance_requires_exact_nonblank_bounded_utf8_fields",
    "tests.test_repair.RepairTests.test_quantization_only_metadata_may_differ_without_requiring_table_order",
    "tests.test_repair.RepairTests.test_reference_quantization_and_changed_input_identity_are_refused",
    "tests.test_repair.RepairTests.test_selected_matrix_encoding_rank_and_both_inputs_finiteness_are_checked",
    "tests.test_repair.RepairTests.test_selection_requires_explicit_unique_present_supported_matrix_names",
    "tests.test_repair.RepairTests.test_semantically_valid_forged_source_or_tensor_binding_cannot_write",
    "tests.test_repair.RepairTests.test_unquantized_tensors_must_preserve_encoding_and_payload_exactly"
]
FIXTURES = [
    {"id": "stories15m-reference", "name": "stories15M.gguf", "bytes": 98357920,
     "sha256": "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a",
     "url": "https://huggingface.co/ggml-org/models-moved/resolve/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M.gguf"},
    {"id": "stories15m-q4_0", "name": "stories15M-q4_0.gguf", "bytes": 19077344,
     "sha256": "6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04",
     "url": "https://huggingface.co/ggml-org/models-moved/resolve/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M-q4_0.gguf"},
]
DOWNLOAD_TOTAL = 117435264
STORAGE_LIMIT = 512 * 1024**2
STREAM_LIMIT, ZIP_LIMIT, ALL_ZIP_LIMIT, REPORT_LIMIT = 1024**2, 256 * 1024, 1024**2, 8 * 1024**2
SELECTED_TENSOR = "blk.0.attn_q.weight"
MAX_ELEMENTS, MAX_ADDED_BYTES, RESERVE_BYTES = 82944, 285120, 1073741824
CDN_HOSTS = {"huggingface.co", "cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
             "aws.cdn.hf.co", "us.aws.cdn.hf.co", "us-east-1.aws.cdn.hf.co", "us-west-2.aws.cdn.hf.co",
             "eu-west-3.aws.cdn.hf.co", "ap-southeast-1.aws.cdn.hf.co", "us.gcp.cdn.hf.co",
             "us-east1.us.gcp.cdn.hf.co", "us-central1.us.gcp.cdn.hf.co", "us-west4.us.gcp.cdn.hf.co",
             "europe-west4.us.gcp.cdn.hf.co", "asia-southeast1.us.gcp.cdn.hf.co"}
PROVENANCE = {"model_id": "ggml-org/models-moved/tinyllamas/stories15M",
              "revision": "4cd12190a40543c61bad59a1b8d982dfc48de020", "license": "MIT (upstream declaration)",
              "reference_source": FIXTURES[0]["url"], "quantized_source": FIXTURES[1]["url"]}
sha = lambda data: hashlib.sha256(data).hexdigest()
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
source_commit = sys.argv[2] if len(sys.argv) == 3 and sys.argv[1] == "--source-commit" else None
report = {
    "schemaVersion": "quantopsy.public-gguf-v2-compatibility.v1", "sourceCommit": source_commit,
    "canonicalBaseCommit": BASE_COMMIT, "packetStatus": "failed", "compatibilityAndCandidateVerified": False,
    "productAccepted": False, "behavioralValidation": "not_run", "startedAtUtc": utc(),
    "commands": [], "guards": [], "sourceFiles": [], "downloads": [], "inspections": [],
    "bindings": [], "errors": [], "storageSamples": [], "generatedFiles": [],
    "scope": {"tier": "tier2", "layer": "fixture-and-real-public-file-cli-integration", "files": FIXTURES,
              "fixtureModules": FIXTURE_MODULES, "fixtureExpected": 92, "legacyFixtureCases": 64,
              "newIntegrationCases": 28, "oldDiagnosticCasesRepeated": False,
              "profile": "v2", "selectedTensor": SELECTED_TENSOR,
              "inference": False, "modelCode": False, "dependencies": False,
              "publicArtifactDiagnose": False, "publicArtifactProbe": False},
    "limits": {"jobMinutes": 15, "workSeconds": 600, "cleanupSeconds": 90, "marginSeconds": 30,
               "downloadSeconds": 120, "redirectsPerFile": 5, "downloadBytesTotal": DOWNLOAD_TOTAL,
               "logicalStorageBytes": STORAGE_LIMIT, "streamBytes": STREAM_LIMIT, "streamGzipBytes": ZIP_LIMIT,
               "allStreamsGzipBytes": ALL_ZIP_LIMIT, "reportBytes": REPORT_LIMIT, "stdoutSeconds": 2,
               "selectedElements": MAX_ELEMENTS, "maxAddedWeightBytes": MAX_ADDED_BYTES, "writerReserveBytes": RESERVE_BYTES},
    "proofLimits": [
        "A verified v2 container/pair and hash-bound candidate are byte-level evidence, not model inference, recovered behavior or full product acceptance.",
        "Declared public MIT/revision/source provenance is not cryptographic lineage or a legal determination.",
        "Only exact two public pinned inputs are downloaded. No tokenizer arrays, weights or raw GGUF metadata region are retained in this report.",
        "Q8_0 is opaque and unselected. Its payload is hashed/copied using the pinned block contract, never numerically decoded.",
        "Selected work/growth budgets are source-derived requests; actual hashes and complete candidate size are measured separately.",
        "Signed redirect queries remain memory-only. Literal HTTPS host admission/public DNS checks are not a host-wide network attestation.",
        "Filesystem identities and hashes are boundary checks, not a filesystem snapshot. Kernel termination can prevent verified cleanup.",
    ],
}
repo = Path.cwd().resolve()
root = None
identity = marker_identity = None
nonce = None
env = {}
baseline = {}
groups = {}
processes = []
attributed = {}
zip_total = 0
source_vector = base_tree = None
base_agents = None
interrupted = False
work_end = cleanup_end = None
tracked_files = {}
known_files = {}
known_directories = {}
copied_helper = None

def need(condition, code):
    if not condition:
        raise RuntimeError(code)


def finite(error):
    value = str(error)
    return value if value and len(value) <= 100 and all(c.islower() or c.isdigit() or c == "_" for c in value) else "observation_error"


def read(file, limit=16 * 1024**2):
    file = Path(file)
    before = file.lstat()
    need(stat.S_ISREG(before.st_mode) and not file.is_symlink() and before.st_size <= limit, "regular_file_bound")
    data = file.read_bytes()
    after = file.lstat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(fields(before) == fields(after) and len(data) == before.st_size, "file_changed")
    return data


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def constant(_):
        raise RuntimeError("nonfinite_json")
    def check(value, depth=0):
        need(depth <= 64, "json_depth")
        if isinstance(value, dict):
            for k, v in value.items():
                k.encode("utf-8")
                check(v, depth + 1)
        elif isinstance(value, list):
            for v in value:
                check(v, depth + 1)
        elif isinstance(value, float):
            need(math.isfinite(value), "nonfinite_json")
        elif isinstance(value, str):
            value.encode("utf-8")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    check(value)
    return value


def owner():
    need(root is not None and identity is not None and marker_identity is not None, "scratch_not_owned")
    current = root.lstat()
    marker = (root / ".owner").lstat()
    need(stat.S_ISDIR(current.st_mode) and not root.is_symlink() and current.st_uid == os.getuid()
         and stat.S_IMODE(current.st_mode) == 0o700 and (current.st_dev, current.st_ino) == identity
         and root.resolve() == root, "scratch_identity_changed")
    need(stat.S_ISREG(marker.st_mode) and not (root / ".owner").is_symlink() and marker.st_uid == os.getuid()
         and stat.S_IMODE(marker.st_mode) == 0o600 and (marker.st_dev, marker.st_ino) == marker_identity
         and read(root / ".owner", 32) == nonce.encode(), "scratch_nonce_changed")


def storage_sample(label):
    owner()
    total = files = 0
    for parent, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            p = Path(parent) / name
            s = p.lstat()
            need(not p.is_symlink() and s.st_uid == os.getuid(), "unexpected_scratch_entry")
            need(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode), "unexpected_scratch_type")
            if stat.S_ISREG(s.st_mode):
                total += s.st_size
                files += 1
    report["storageSamples"].append({"label": label, "logicalBytes": total, "files": files})
    need(total <= STORAGE_LIMIT and files <= 64, "scratch_storage_bound")


def proc_info(pid):
    folder = Path("/proc") / str(pid)
    uid = folder.stat().st_uid
    before = (folder / "stat").read_text()
    close = before.rfind(")")
    need(close > 0, "proc_stat_shape")
    fields = before[close + 2:].split()
    need(len(fields) >= 20 and fields[19].isdigit(), "proc_stat_shape")
    after = (folder / "stat").read_text()
    need(after[after.rfind(")") + 2:].split()[19] == fields[19], "proc_identity_changed")
    return {"pid": pid, "startTime": fields[19], "pgid": int(fields[2]), "state": fields[0], "uid": uid}


def owned():
    rows, uncertain = [], []
    entries = list(Path("/proc").iterdir())
    need(len(entries) <= 8192, "proc_inventory_bound")
    for folder in entries:
        if not folder.name.isdigit():
            continue
        pid = int(folder.name)
        if pid == os.getpid():
            continue
        try:
            item = proc_info(pid)
            if item["uid"] != os.getuid() or baseline.get(pid) == item["startTime"]:
                continue
            cwd = Path(os.readlink(folder / "cwd"))
            exe = Path(os.readlink(folder / "exe"))
            original = False
            if item["pgid"] in groups:
                try:
                    leader = proc_info(item["pgid"])
                    if leader["startTime"] != groups[item["pgid"]]:
                        uncertain.append(pid)
                        continue
                    original = True
                except (FileNotFoundError, ProcessLookupError):
                    pass
            scoped = cwd == root or root in cwd.parents or root in exe.parents
            if original or scoped:
                need(proc_info(pid)["startTime"] == item["startTime"], "proc_identity_changed")
                rows.append(item)
                attributed[(pid, item["startTime"])] = item
            elif item["pgid"] in groups:
                uncertain.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError, RuntimeError, IndexError):
            uncertain.append(pid)
    need(len(rows) <= 128 and len(uncertain) <= 128 and len(attributed) <= 256, "owned_inventory_bound")
    return rows, uncertain


def group_gone(pid):
    try:
        os.killpg(pid, 0)
        return False
    except ProcessLookupError:
        return True


def signal_original_group(pid, number):
    if group_gone(pid):
        return False
    try:
        current = proc_info(pid)
        need(current["uid"] == os.getuid() and current["startTime"] == groups[pid], "signal_identity_refused")
        os.killpg(pid, number)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def capture(data, observed, digest, complete):
    global zip_total
    zipped = gzip.compress(data, mtime=0)
    result = {"encoding": "gzip-base64", "bytes": len(data), "sha256": sha(data),
              "observedBytes": observed, "observedSha256": digest,
              "complete": complete and observed == len(data)}
    if len(zipped) <= ZIP_LIMIT and zip_total + len(zipped) <= ALL_ZIP_LIMIT:
        zip_total += len(zipped)
        result.update({"compressedBytes": len(zipped), "data": base64.b64encode(zipped).decode("ascii"), "retained": True})
    else:
        result.update({"complete": False, "retained": False})
    return result


def run(label, argv, seconds, allowed=(0,), cwd=None, cleanup=False, post_scratch=False):
    if post_scratch:
        need(cleanup and report.get("scratchCleanup") == "passed" and not root.exists(), "post_scratch_guard")
    else:
        owner()
    need(cleanup or not interrupted, "interrupted_before_command")
    end = cleanup_end if cleanup else work_end
    need(end is not None and end - time.monotonic() >= 1, "command_budget_exhausted")
    started = time.monotonic()
    deadline = min(end - 0.5, started + seconds)
    command_env = env if not post_scratch else {k: v for k, v in env.items() if k in
                  {"PATH", "HOME", "LANG", "LC_ALL", "TZ", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT", "GIT_OPTIONAL_LOCKS"}}
    process = subprocess.Popen(argv, cwd=cwd or repo, env=command_env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    processes.append(process)
    row = {"label": label, "argv": [str(a).replace(str(root), "[scratch]").replace(str(repo), "[source]") for a in argv],
           "pid": process.pid, "startTime": None, "exitCode": None, "timedOut": False,
           "termSent": False, "killSent": False, "ownershipError": None}
    report["commands"].append(row)
    selector = selectors.DefaultSelector()
    pipes = {name: {"data": bytearray(), "digest": hashlib.sha256(), "observed": 0, "eof": False}
             for name in ("stdout", "stderr")}
    stopped = leader_exited = None
    last_scan = last_storage = started
    try:
        info = proc_info(process.pid)
        need(info["uid"] == os.getuid() and info["pgid"] == process.pid, "child_group_identity")
        groups[process.pid] = info["startTime"]
        row["startTime"] = info["startTime"]
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while True:
            now = time.monotonic()
            code = process.poll()
            if code is not None and leader_exited is None:
                leader_exited = now
            absent = group_gone(process.pid)
            if code is not None and not selector.get_map() and absent:
                break
            overflow = any(v["observed"] > STREAM_LIMIT for v in pipes.values())
            if stopped is None and ((interrupted and not cleanup) or overflow or now >= deadline or
                                    (leader_exited is not None and not absent and now - leader_exited > 1.5)):
                row["timedOut"] = now >= deadline
                row["stopReason"] = "interrupted" if interrupted else "output_limit" if overflow else "timeout" if row["timedOut"] else "group_remaining"
                stopped = now
                try:
                    row["termSent"] = signal_original_group(process.pid, signal.SIGTERM)
                except (OSError, RuntimeError):
                    row["ownershipError"] = "signal_identity_refused"
            if stopped is not None and now >= stopped + 3 and not row["killSent"] and not absent:
                try:
                    row["killSent"] = signal_original_group(process.pid, signal.SIGKILL)
                except (OSError, RuntimeError):
                    row["ownershipError"] = "signal_identity_refused"
            if stopped is not None and now >= stopped + 6.5:
                break
            for key, _ in selector.select(0.05):
                chunk = os.read(key.fileobj.fileno(), 65536)
                state = pipes[key.data]
                if not chunk:
                    state["eof"] = True
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                state["observed"] += len(chunk)
                state["digest"].update(chunk)
                state["data"].extend(chunk[:max(0, STREAM_LIMIT - len(state["data"]))])
            if now - last_scan >= 0.5:
                owned()
                last_scan = now
            if not post_scratch and now - last_storage >= 2:
                storage_sample(label)
                last_storage = now
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
        row.update({"durationMs": round((time.monotonic() - started) * 1000), "exitCode": process.returncode,
                    "exitObservedAndReaped": process.returncode is not None,
                    "originalProcessGroupAbsent": group_gone(process.pid),
                    "descriptorsClosed": [process.stdout.closed, process.stderr.closed],
                    "pipesEOF": [pipes[n]["eof"] for n in ("stdout", "stderr")]})
        for name, state in pipes.items():
            row[name] = capture(bytes(state["data"]), state["observed"], state["digest"].hexdigest(), state["eof"])
    row["normallyClosed"] = (row["exitObservedAndReaped"] and row["originalProcessGroupAbsent"] and
                            all(row["descriptorsClosed"]) and all(row["pipesEOF"]) and not row["timedOut"] and
                            not row["termSent"] and not row["killSent"] and row["ownershipError"] is None and
                            row.get("stopReason") is None and
                            all(row[n]["complete"] and row[n]["retained"] for n in ("stdout", "stderr")))
    need(row["normallyClosed"], "command_lifecycle_incomplete")
    need(row["exitCode"] in allowed, "unexpected_child_exit")
    return bytes(pipes["stdout"]["data"]), bytes(pipes["stderr"]["data"]), row


def parse_git(data, index=False, sizes=False):
    rows = {}
    for record in data.decode("utf-8").split("\0"):
        if not record:
            continue
        header, name = record.split("\t", 1)
        parts = header.split()
        need(name not in rows and not name.startswith("/") and all(p not in ("", ".", "..") for p in name.split("/")), "git_path_invalid")
        need(parts[2] == "0" if index else parts[1] == "blob", "git_entry_invalid")
        rows[name] = {"mode": parts[0], "gitBlob": parts[1] if index else parts[2]}
        if sizes:
            rows[name]["bytes"] = int(parts[3])
    return rows


def file_vector(name):
    file = repo / name
    for parent in file.parents:
        if parent == repo:
            break
        need(parent.is_dir() and not parent.is_symlink(), "source_ancestor_changed")
    info = file.lstat()
    if file.is_symlink():
        mode, data = "120000", os.fsencode(os.readlink(file))
    else:
        need(stat.S_ISREG(info.st_mode), "source_file_type")
        mode, data = ("100755" if info.st_mode & 0o111 else "100644"), read(file)
    return {"path": name, "mode": mode, "bytes": len(data), "sha256": sha(data),
            "gitBlob": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}


def file_identity(info):
    return {"device": str(info.st_dev), "inode": str(info.st_ino), "uid": info.st_uid,
            "mode": format(stat.S_IMODE(info.st_mode), "04o"), "bytes": info.st_size,
            "mtimeNs": str(info.st_mtime_ns), "ctimeNs": str(info.st_ctime_ns)}


def fingerprint(file, expected=None):
    owner()
    file = Path(file)
    need(root in file.parents and not file.is_symlink(), "input_outside_owned_root")
    for parent in file.parents:
        if parent == root:
            break
        need(parent.is_dir() and not parent.is_symlink(), "input_ancestor_changed")
    before = file.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
         and stat.S_IMODE(before.st_mode) == 0o600 and before.st_size <= DOWNLOAD_TOTAL, "input_type_or_bound")
    fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        need(file_identity(os.fstat(fd)) == file_identity(before), "input_changed_at_open")
        remaining = before.st_size
        while remaining:
            need(time.monotonic() < (cleanup_end if cleanup_end is not None else work_end), "hash_deadline")
            chunk = os.read(fd, min(1024**2, remaining))
            need(bool(chunk), "input_truncated")
            remaining -= len(chunk)
            digest.update(chunk)
        need(os.read(fd, 1) == b"" and file_identity(file.lstat()) == file_identity(before)
             == file_identity(os.fstat(fd)), "input_changed_during_hash")
    finally:
        os.close(fd)
    result = {"relativePath": str(file.relative_to(root)), **file_identity(before), "sha256": digest.hexdigest()}
    if expected is not None:
        need(result["bytes"] == expected["bytes"] and result["sha256"] == expected["sha256"], "input_identity_mismatch")
    return result


def verify_inputs(label):
    result = []
    for fixture in FIXTURES:
        file = root / "models" / fixture["name"]
        if fixture["id"] in tracked_files:
            current = fingerprint(file, fixture)
            need(current == tracked_files[fixture["id"]], "input_changed_between_phases")
            result.append({"fixture": fixture["id"], **current})
    report.setdefault("inputChecks", []).append({"label": label, "files": result})
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def url_record(url):
    value = urllib.parse.urlsplit(url)
    return {"host": value.hostname, "path": value.path, "queryRedacted": bool(value.query)}


def validate_url(url, initial=False):
    need(isinstance(url, str) and len(url) <= 16384 and not any(ord(c) < 33 or ord(c) == 127 for c in url), "download_url_bound")
    value = urllib.parse.urlsplit(url)
    need(value.scheme == "https" and value.hostname in CDN_HOSTS and value.port in (None, 443)
         and not value.username and not value.password and not value.fragment, "download_url_refused")
    path = urllib.parse.unquote(value.path)
    need(path.startswith("/") and len(path) <= 4096 and "\\" not in path
         and all(p not in (".", "..") for p in path.split("/"))
         and not any(ord(c) < 32 or ord(c) == 127 for c in path), "download_path_refused")
    if initial:
        need(url in [f["url"] for f in FIXTURES] and not value.query, "immutable_download_source_required")
    addresses = socket.getaddrinfo(value.hostname, 443, type=socket.SOCK_STREAM)
    need(1 <= len(addresses) <= 32 and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses), "download_dns_not_public")
    return value


def download(fixture):
    owner()
    row = {"fixture": fixture["id"], "source": fixture["url"], "expectedBytes": fixture["bytes"],
           "expectedSha256": fixture["sha256"], "redirects": [], "receivedBytes": 0, "writtenBytes": 0, "responseObtained": False,
           "complete": False, "responseClosed": False, "fileDescriptorClosed": False}
    report["downloads"].append(row)
    target, temporary = root / "models" / fixture["name"], root / "tmp" / (fixture["name"] + ".partial")
    need(not os.path.lexists(target) and not os.path.lexists(temporary), "download_output_exists")
    started = time.monotonic()
    duration = min(120, max(0, work_end - started - 1))
    need(duration > 0 and not interrupted, "download_budget_exhausted")
    fd = created = response = previous = None
    alarm_installed = False
    digest = hashlib.sha256()
    def expired(_number, _frame):
        raise RuntimeError("download_deadline")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        created = os.fstat(fd)
        need((temporary.lstat().st_dev, temporary.lstat().st_ino) == (created.st_dev, created.st_ino), "download_file_replaced")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, duration)
        alarm_installed = True
        context = ssl.create_default_context()
        need(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED and context.keylog_filename is None, "verified_tls_required")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                            urllib.request.HTTPSHandler(context=context))
        url = fixture["url"]
        for hop in range(6):
            need(not interrupted and time.monotonic() < work_end, "download_interrupted_or_expired")
            validate_url(url, initial=hop == 0)
            request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "Quantopsy-public-observation/1"})
            try:
                response = opener.open(request, timeout=min(10, max(0.2, duration - (time.monotonic() - started))))
            except urllib.error.HTTPError as error:
                response = error
            row["responseObtained"] = True
            status = response.code
            if status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location")
                need(hop < 5 and isinstance(location, str) and len(location) <= 16384, "redirect_limit_or_location")
                next_url = urllib.parse.urljoin(url, location)
                validate_url(next_url)
                row["redirects"].append({"status": status, "from": url_record(url), "to": url_record(next_url)})
                response.close()
                row["responseClosed"] = True
                response = None
                url = next_url
                continue
            row["httpStatus"] = status
            row["terminal"] = url_record(url)
            need(status == 200, "download_http_refusal")
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            need(encoding in ("", "identity"), "download_content_encoding_refused")
            declared = response.headers.get("Content-Length")
            need(declared is not None and declared.isascii() and declared.isdigit()
                 and int(declared) == fixture["bytes"], "download_content_length_mismatch")
            row["declaredBytes"] = int(declared)
            while True:
                need(not interrupted and time.monotonic() - started < 120 and time.monotonic() < work_end, "download_deadline")
                chunk = response.read(min(1024**2, fixture["bytes"] - row["receivedBytes"]) or 1)
                if not chunk:
                    row["bodyEOFObserved"] = True
                    break
                need(row["receivedBytes"] + len(chunk) <= fixture["bytes"], "download_byte_bound")
                need(sum(x["receivedBytes"] for x in report["downloads"]) + len(chunk) <= DOWNLOAD_TOTAL, "download_total_byte_bound")
                row["receivedBytes"] += len(chunk)
                offset = 0
                while offset < len(chunk):
                    written = os.write(fd, chunk[offset:])
                    need(written > 0, "download_write_incomplete")
                    digest.update(chunk[offset:offset + written])
                    row["writtenBytes"] += written
                    offset += written
                if row["receivedBytes"] % (8 * 1024**2) < len(chunk):
                    storage_sample("download_" + fixture["id"])
            need(row["receivedBytes"] == row["writtenBytes"] == fixture["bytes"] and digest.hexdigest() == fixture["sha256"], "download_identity_mismatch")
            os.fsync(fd)
            os.close(fd)
            fd = None
            need(file_identity(temporary.lstat())["device"] == str(created.st_dev)
                 and temporary.lstat().st_ino == created.st_ino, "download_file_replaced")
            os.link(temporary, target, follow_symlinks=False)
            os.unlink(temporary)
            current = fingerprint(target, fixture)
            tracked_files[fixture["id"]] = current
            row.update({"complete": True, "sha256": digest.hexdigest(), "file": current})
            break
        need(row["complete"], "download_nonterminal")
    except urllib.error.URLError:
        row["error"] = "download_transport_error"
        raise RuntimeError("download_transport_error")
    except Exception as error:
        row["error"] = finite(error)
        raise
    finally:
        row["resourceErrors"] = []
        if alarm_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except (OSError, ValueError):
                row["resourceErrors"].append("download_alarm_cancel_failed")
        if previous is not None:
            try:
                signal.signal(signal.SIGALRM, previous)
            except (OSError, ValueError):
                row["resourceErrors"].append("download_handler_restore_failed")
        try:
            if response is not None:
                response.close()
            row["responseClosed"] = True
        except OSError:
            row["responseClosed"] = False
            row["resourceErrors"].append("download_response_close_failed")
        try:
            if fd is not None:
                os.close(fd)
            row["fileDescriptorClosed"] = True
        except OSError:
            row["fileDescriptorClosed"] = False
            row["resourceErrors"].append("download_descriptor_close_failed")
        row["writtenSha256"] = digest.hexdigest()
        row["durationMs"] = round((time.monotonic() - started) * 1000)
        if created is not None and os.path.lexists(temporary):
            row["partialFile"] = {"relativePath": str(temporary.relative_to(root)),
                                  "createdDevice": str(created.st_dev), "createdInode": str(created.st_ino),
                                  "writtenBytes": row["writtenBytes"], "writtenSha256": digest.hexdigest(),
                                  "meaning": "Digest of bytes written to the originally acquired descriptor; current path is independently checked during cleanup."}
    storage_sample("download_complete_" + fixture["id"])


def canonical_base():
    global base_tree, base_agents
    stdout, _, _ = run("canonical_parent", ["git", "rev-parse", "HEAD^"], 10)
    need(stdout.strip().decode() == BASE_COMMIT, "canonical_parent_changed")
    stdout, _, _ = run("canonical_tree", ["git", "ls-tree", "-r", "-l", "-z", BASE_COMMIT], 10)
    base_tree = parse_git(stdout, sizes=True)
    need(len(base_tree) == 41 and sum(x["bytes"] for x in base_tree.values()) == BASE_BYTES
         and not any(n in base_tree for n in ADDED) and set(MODIFIED) <= set(base_tree)
         and base_tree.get(ACTIVE) == {"mode": "100644", "gitBlob": BASE_STUB_BLOB, "bytes": BASE_STUB_BYTES},
         "canonical_baseline_changed")
    base_agents, _, _ = run("canonical_agents", ["git", "cat-file", "blob", base_tree["AGENTS.md"]["gitBlob"]], 10)
    report["canonicalBaseline"] = {"commit": BASE_COMMIT, "blobs": 41, "bytes": BASE_BYTES,
                                   "onlyModifiedPriorPaths": MODIFIED, "onlyAddedPaths": ADDED,
                                   "allUnmodifiedPriorPathsRequiredUnchanged": True,
                                   "inertDispatchStub": {"path": ACTIVE, "gitBlob": BASE_STUB_BLOB,
                                                        "bytes": BASE_STUB_BYTES, "sha256": BASE_STUB_SHA256,
                                                        "requiredUnchangedInPreparedSource": True}}

def guard(label, cleanup=False, post_scratch=False):
    global source_vector
    item = {"label": label, "passed": False}
    report["guards"].append(item)
    options = {"cleanup": cleanup, "post_scratch": post_scratch}
    out, _, _ = run(label + "_head", ["git", "rev-parse", "HEAD"], 10, **options)
    need(out.strip().decode() == source_commit and sha(read(repo / ACTIVE)) == BASE_STUB_SHA256,
         "source_head_or_inert_workflow_changed")
    tree, _, _ = run(label + "_tree", ["git", "ls-tree", "-r", "-z", "HEAD"], 10, **options)
    index, _, _ = run(label + "_index", ["git", "ls-files", "--stage", "-z"], 10, **options)
    tree, index = parse_git(tree), parse_git(index, index=True)
    need(base_tree is not None and len(tree) == len(index) == 49
         and set(base_tree) <= set(tree), "source_path_count")
    need(sorted(set(tree) - set(base_tree)) == ADDED, "source_added_paths")
    for name, previous in base_tree.items():
        need(tree[name]["mode"] == previous["mode"], "prior_source_mode_changed")
        if name in MODIFIED:
            current = read(repo / name)
            need(len(current) == MODIFIED[name]["bytes"] and sha(current) == MODIFIED[name]["sha256"],
                 "scoped_modified_source_changed")
            if name == "AGENTS.md":
                need(current.startswith(base_agents) and len(current) > len(base_agents), "scope_prefix_changed")
        else:
            need(tree[name]["gitBlob"] == previous["gitBlob"], "unmodified_prior_source_blob_changed")
    for name, expected in NEW_PINS.items():
        need(tree[name]["mode"] == "100644" and len(read(repo / name)) == expected["bytes"]
             and sha(read(repo / name)) == expected["sha256"], "new_fixture_or_helper_changed")
    vector = []
    for name, record in tree.items():
        actual = file_vector(name)
        need(index[name] == record and actual["mode"] == record["mode"] and actual["gitBlob"] == record["gitBlob"],
             "source_index_or_body_changed")
        vector.append(actual)
    for suffix, args in (("status", ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]),
                         ("ignored", ["git", "status", "--porcelain=v1", "-z", "--ignored=matching", "--untracked-files=normal"])):
        out, _, _ = run(label + "_" + suffix, args, 10, **options)
        need(out == b"", "source_generated_or_dirty_output")
    run(label + "_diff", ["git", "-c", "core.filemode=true", "diff", "--no-ext-diff", "--no-textconv",
                         "--quiet", "--exit-code", "HEAD", "--"], 10, **options)
    if source_vector is None:
        source_vector = vector
        report["sourceFiles"] = vector
    need(vector == source_vector, "source_vector_changed")
    item.update({"passed": True, "blobs": len(vector), "bytes": sum(x["bytes"] for x in vector),
                 "vectorSha256": sha(json.dumps(vector, ensure_ascii=False, separators=(",", ":")).encode()),
                 "fullGitIndexNativeBindings": True, "allUnmodifiedPriorPathsUnchanged": True,
                 "scopedPriorModifications": sorted(MODIFIED)})

def canonical(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()

def owned_write(relative, data):
    owner()
    need(isinstance(data, bytes) and len(data) <= 2 * 1024**2 and relative not in known_files, "generated_file_arguments")
    target = root / relative
    need(target.parent.is_dir() and str(target.parent.relative_to(root)) in known_directories
         and not os.path.lexists(target), "generated_file_not_absent")
    fd = None
    created = None
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        created = os.fstat(fd)
        known_files[relative] = {"createdDevice": str(created.st_dev), "createdInode": str(created.st_ino), "file": None}
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        current = fingerprint(target, {"bytes": len(data), "sha256": sha(data)})
        need(current["device"] == str(created.st_dev) and current["inode"] == str(created.st_ino), "generated_file_replaced")
        known_files[relative]["file"] = current
        report["generatedFiles"].append(current)
        return current
    finally:
        if fd is not None:
            os.close(fd)
        if created is not None and known_files[relative]["file"] is None and os.path.lexists(target):
            current = fingerprint(target)
            need(current["device"] == str(created.st_dev) and current["inode"] == str(created.st_ino), "partial_generated_file_replaced")
            known_files[relative]["file"] = current

def collect_outputs():
    owner()
    directory = root / "output/candidate"
    if os.path.lexists(directory):
        info = directory.lstat()
        need(stat.S_ISDIR(info.st_mode) and not directory.is_symlink() and info.st_uid == os.getuid()
             and stat.S_IMODE(info.st_mode) == 0o700, "candidate_directory_type")
        current = file_identity(info)
        if "output/candidate" in known_directories:
            need(current["device"] == known_directories["output/candidate"]["device"]
                 and current["inode"] == known_directories["output/candidate"]["inode"], "candidate_directory_replaced")
        else:
            known_directories["output/candidate"] = current
    for relative in ("output/repair-plan-v2.json", "output/candidate/candidate.gguf",
                     "output/candidate/plan.json", "output/candidate/repair.json"):
        path = root / relative
        if os.path.lexists(path):
            current = fingerprint(path)
            if relative in known_files:
                need(current == known_files[relative]["file"], "generated_output_changed")
            else:
                known_files[relative] = {"createdDevice": current["device"], "createdInode": current["inode"], "file": current}
                report["generatedFiles"].append(current)

def verify_generated(label):
    checks = []
    for relative, expected in known_files.items():
        need(expected["file"] is not None, "generated_file_identity_unavailable")
        current = fingerprint(root / relative)
        need(current == expected["file"], "generated_file_changed_between_phases")
        checks.append(current)
    report.setdefault("generatedChecks", []).append({"label": label, "files": checks})

SCALAR_KEYS = {"general.architecture", "general.name", "general.file_type", "general.quantization_version",
               "general.alignment", "llama.block_count", "llama.embedding_length", "llama.feed_forward_length",
               "llama.attention.head_count", "llama.attention.head_count_kv", "llama.context_length",
               "llama.rope.dimension_count", "llama.attention.layer_norm_rms_epsilon"}
QUANTIZATION_KEYS = {"general.file_type", "general.quantization_version", "general.quantized_by", "general.quantized_at"}

def hash_span(file, offset, count):
    need(type(offset) is int and type(count) is int and offset >= 0 and count >= 0, "hash_span_arguments")
    before = file.lstat()
    need(offset + count <= before.st_size, "hash_span_outside_file")
    fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        need(file_identity(os.fstat(fd)) == file_identity(before), "hash_span_file_changed")
        position, remaining = offset, count
        while remaining:
            need(not interrupted and time.monotonic() < work_end, "hash_span_deadline")
            data = os.pread(fd, min(1024**2, remaining), position)
            need(bool(data), "hash_span_truncated")
            digest.update(data)
            position += len(data)
            remaining -= len(data)
        need(file_identity(os.fstat(fd)) == file_identity(before) == file_identity(file.lstat()), "hash_span_file_changed")
    finally:
        os.close(fd)
    return digest.hexdigest()


def decoded_original(value, maximum, encoding):
    need(value.get("encoding") == encoding and type(value.get("bytes")) is int and 0 <= value["bytes"] <= maximum, "descriptor_original_shape")
    raw = base64.b64decode(value["data"], validate=True)
    need(base64.b64encode(raw).decode() == value["data"], "descriptor_base64_not_canonical")
    if encoding == "gzip-base64":
        need(len(raw) == value["compressedBytes"] and len(raw) <= maximum + 1024, "descriptor_gzip_bound")
        decoder = zlib.decompressobj(31)
        decoded = decoder.decompress(raw, maximum + 1)
        need(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data and len(decoded) <= maximum, "descriptor_gzip_incomplete")
        raw = decoded
    need(len(raw) == value["bytes"] and sha(raw) == value["sha256"], "descriptor_original_digest")
    return raw


def tensor_bytes(tensor):
    dimensions, kind = tensor["dimensions"], tensor["ggmlType"]
    elements = math.prod(dimensions)
    need(kind in (0, 1, 2, 8), "unknown_encoding_no_byte_guess")
    if kind in (2, 8):
        need(dimensions[0] % 32 == 0, "quantized_row_block_shape")
        return elements // 32 * (18 if kind == 2 else 34)
    return elements * (4 if kind == 0 else 2)

def validate_inventory(file, file_bytes, value):
    header = value["header"]
    raw_header = decoded_original(header["raw"], 24, "base64")
    need(len(raw_header) == 24 and header["magicHex"] == "47475546" and header["version3LittleEndianObserved"] is True, "inventory_header")
    magic, version, tensors, metadata = struct.unpack("<4sIQQ", raw_header)
    need(magic == b"GGUF" and version == header["littleEndianVersion"] == 3 and tensors == header["tensorCount"]
         and metadata == header["metadataCount"] and 1 <= tensors <= 8192 and metadata <= 4096, "inventory_header_counts")
    need(hash_span(file, 0, 24) == sha(raw_header), "header_input_binding")
    rows, position = value["metadata"], 24
    need(len(rows) == metadata and len({x["key"] for x in rows}) == metadata, "metadata_inventory_count")
    scalars = {}
    for ordinal, row in enumerate(rows):
        need(row["ordinal"] == ordinal and row["offset"] == position and type(row["encodedBytes"]) is int
             and row["encodedBytes"] > 0 and position + row["encodedBytes"] <= file_bytes
             and hash_span(file, position, row["encodedBytes"]) == row["encodedSha256"], "metadata_encoding_hash_binding")
        position += row["encodedBytes"]
        need(type(row["kind"]) is int and row["kind"] in range(13) and type(row["count"]) is int
             and 0 <= row["count"] <= 262144 and isinstance(row["key"], str) and len(row["key"].encode()) <= 1024, "metadata_entry_shape")
        if row["valueRetained"]:
            need(row["key"] in SCALAR_KEYS and row["kind"] != 9 and "scalarValue" in row
                 and row["scalarRepresentation"] in ("json", "decimal-string"), "metadata_value_retention_refused")
            scalar = row["scalarValue"]
            need(type(scalar) in (str, int, float, bool) and (not isinstance(scalar, str) or
                 len(scalar.encode()) <= 2048 and not any(ord(c) < 32 or ord(c) == 127 for c in scalar)), "retained_scalar_bound")
            scalars[row["key"]] = {"kind": row["kind"], "value": scalar}
        else:
            need("scalarValue" not in row and "scalarRepresentation" not in row, "unretained_metadata_value")
    region = value["metadataRegion"]
    need(region["offset"] == 24 and region["bytes"] == position - 24 <= 32 * 1024**2
         and region["rawRetained"] is False and hash_span(file, 24, position - 24) == region["sha256"]
         and value["allowlistedScalars"] == scalars, "complete_metadata_inventory_binding")
    descriptor = value["descriptorRegion"]
    raw = decoded_original(descriptor["raw"], 256 * 1024, "gzip-base64")
    need(descriptor["offset"] == position and descriptor["bytes"] == len(raw) and descriptor["sha256"] == sha(raw)
         and hash_span(file, position, len(raw)) == sha(raw), "descriptor_input_binding")
    cursor = 0
    def take(count):
        nonlocal cursor
        need(type(count) is int and 0 <= count <= len(raw) - cursor, "descriptor_parse_bound")
        data = raw[cursor:cursor + count]
        cursor += count
        return data
    def number(code):
        return struct.unpack("<" + code, take(struct.calcsize("<" + code)))[0]
    observed = value["tensors"]
    need(len(observed) == tensors, "complete_tensor_inventory_count")
    alignment = value["tensorData"]["alignment"]
    need(type(alignment) is int and 8 <= alignment <= 4096 and alignment % 8 == 0, "inventory_alignment")
    metadata_alignment = scalars.get("general.alignment")
    if any(row["key"] == "general.alignment" for row in rows):
        need(metadata_alignment is not None and metadata_alignment["kind"] == 4
             and type(metadata_alignment["value"]) is int and metadata_alignment["value"] == alignment, "present_alignment_not_supported")
    else:
        need(alignment == 32, "default_alignment_not_32")
    data_start = position + len(raw) + (-(position + len(raw)) % alignment)
    for ordinal, tensor in enumerate(observed):
        begin = cursor
        length = number("Q")
        need(0 < length <= 64, "tensor_name_bound")
        name = take(length).decode("utf-8")
        rank = number("I")
        need(1 <= rank <= 4, "tensor_rank")
        dimensions = [number("Q") for _ in range(rank)]
        kind, offset = number("I"), number("Q")
        need(tensor["ordinal"] == ordinal and tensor["name"] == name and tensor["dimensions"] == dimensions
             and all(x > 0 for x in dimensions) and tensor["ggmlType"] == kind and tensor["relativeOffset"] == offset
             and offset % alignment == 0 and tensor["absoluteDataOffset"] == data_start + offset <= file_bytes
             and tensor["descriptor"] == {"offset": position + begin, "encodedBytes": cursor - begin,
                 "encodedSha256": sha(raw[begin:cursor])}, "ordered_tensor_descriptor_binding")
    need(cursor == len(raw) and len({t["name"] for t in observed}) == tensors, "descriptor_inventory_complete")
    need(value["tensorData"]["offset"] == data_start and value["tensorData"]["payloadBytesReadForDecoding"] == 0
         and value["tensorData"]["payloadLengthsInferred"] is False, "observer_payload_limit")
    need(value["payloadEvidence"] == {"allSpansValidated": True, "allPayloadsHashed": True,
         "payloadsDecoded": False, "sizeSource": "explicit_ggml_0_1_2_8_blocks"}, "payload_evidence_contract")
    spans = []
    for tensor in observed:
        count = tensor_bytes(tensor)
        need(tensor["elements"] == math.prod(tensor["dimensions"]) and tensor["payloadBytes"] == count
             and tensor["absoluteDataOffset"] + count <= file_bytes
             and tensor["payloadSha256"] == hash_span(file, tensor["absoluteDataOffset"], count), "fresh_payload_hash_binding")
        spans.append((tensor["absoluteDataOffset"], tensor["absoluteDataOffset"] + count))
    ordered = sorted(spans)
    need(all(ordered[i - 1][1] <= ordered[i][0] for i in range(1, len(ordered))), "payload_overlap")
    return {"allPayloadHashesBound": True, "headerBound": True, "allMetadataHashesBound": True, "orderedDescriptorsBound": True,
            "metadataEntries": metadata, "tensors": tensors, "numericTensorTypes": dict(collections.Counter(str(t["ggmlType"]) for t in observed))}


def validate_inspection(fixture, inventory, value):
    scalar = lambda key: inventory["allowlistedScalars"].get(key, {}).get("value")
    need(value.get("schema_version") == 2 and value.get("kind") == "quantopsy.container-inspection"
         and value.get("file") == {"name": fixture["name"], "size": fixture["bytes"], "sha256": fixture["sha256"]}
         and value.get("architecture") == scalar("general.architecture") and value.get("model_name") == scalar("general.name")
         and value.get("quantization_version") == scalar("general.quantization_version")
         and value.get("alignment") == inventory["tensorData"]["alignment"]
         and value.get("metadata_entries") == inventory["header"]["metadataCount"]
         and value.get("evidence") == "container_structure_and_bytes" and value.get("behavioral_validation") == "not_run", "application_inspection_binding")
    names = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0"}
    expected = []
    spans = []
    for tensor in inventory["tensors"]:
        count = tensor_bytes(tensor)
        need(tensor["absoluteDataOffset"] + count <= fixture["bytes"], "application_tensor_span")
        spans.append((tensor["absoluteDataOffset"], tensor["absoluteDataOffset"] + count))
        expected.append({"name": tensor["name"], "dimensions": tensor["dimensions"], "type": names[tensor["ggmlType"]],
                         "bytes": count, "elements": math.prod(tensor["dimensions"])})
    need(value.get("tensors") == expected and value.get("tensor_types") == dict(collections.Counter(x["type"] for x in expected)), "complete_application_inventory_binding")
    ordered = sorted(spans)
    need(all(ordered[i - 1][1] <= ordered[i][0] for i in range(1, len(ordered))), "application_overlapping_spans")


FIXTURE_PROGRAM = """
import json, sys, unittest
sys.path.insert(0, sys.argv[1])
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[2:])
def ids(value):
    if isinstance(value, unittest.TestSuite):
        return [name for child in value for name in ids(child)]
    return [value.id()]
identities = sorted(ids(suite))
collected = suite.countTestCases()
result = unittest.TextTestRunner(verbosity=2).run(suite)
summary = {"collected": collected, "testsRun": result.testsRun, "identities": identities,
           "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
           "expectedFailures": len(result.expectedFailures), "unexpectedSuccesses": len(result.unexpectedSuccesses)}
print("QUANTOPSY_V2_FIXTURE_RESULT=" + json.dumps(summary, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
CLI_PROGRAM = "import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=['quantopsy',*sys.argv[2:]];runpy.run_module('quantopsy',run_name='__main__')"

def fixture_checks():
    out, err, row = run("fixture_reader_repair_cli_and_v2", [sys.executable, "-I", "-B", "-S", "-c",
                        FIXTURE_PROGRAM, str(repo), *FIXTURE_MODULES], 120, allowed=(0, 1))
    prefix = b"QUANTOPSY_V2_FIXTURE_RESULT="
    need(out.startswith(prefix) and out.endswith(b"\n") and out.count(prefix) == 1, "fixture_original_marker")
    value = strict_json(out[len(prefix):-1])
    report["fixture"] = {"modules": FIXTURE_MODULES, "expected": 92, "actual": value, "accepted": False,
                         "stdoutCommand": row["label"], "stderrRetained": row["stderr"]["complete"]}
    need(type(value) is dict and set(value) == {"collected", "testsRun", "identities", "failures", "errors",
         "skips", "expectedFailures", "unexpectedSuccesses"} and all(type(value[k]) is int for k in
         ("collected", "testsRun", "failures", "errors", "skips", "expectedFailures", "unexpectedSuccesses")),
         "fixture_summary_shape")
    need(row["exitCode"] == 0 and value["collected"] == value["testsRun"] == 92
         and value["identities"] == FIXTURE_IDS and len(set(value["identities"])) == 92
         and all(value[k] == 0 for k in ("failures", "errors", "skips", "expectedFailures", "unexpectedSuccesses")),
         "fixture_checks_failed")
    report["fixture"]["accepted"] = True

def product_command(label, arguments, seconds=90, outputs=False):
    try:
        out, err, row = run(label, [sys.executable, "-I", "-B", "-S", "-c", CLI_PROGRAM, str(repo), *map(str, arguments)],
                            seconds, allowed=(0, 2), cwd=root / "output")
    finally:
        # Only known, previously absent outputs are admitted after their writer is reaped.
        if outputs and processes and processes[-1].poll() is not None and group_gone(processes[-1].pid):
            collect_outputs()
    entry = {"label": label, "exitCode": row["exitCode"], "accepted": False}
    report.setdefault("applicationCommands", []).append(entry)
    if out:
        entry["stdoutJson"] = strict_json(out)
    if err:
        entry["stderrJson"] = strict_json(err)
    need(row["exitCode"] == 0 and err == b"" and out != b"", "v2_application_refused_or_failed")
    entry["accepted"] = True
    return strict_json(out), out

def inspect_file(label, file, expected):
    value, raw = product_command(label, ["inspect", file, "--profile", "v2"])
    row = {"label": label, "file": expected, "value": value, "boundToIndependentInventory": False}
    report["inspections"].append(row)
    need(value.get("schema_version") == 2 and value.get("kind") == "quantopsy.container-inspection"
         and value.get("file") == {"name": expected["name"], "size": expected["bytes"], "sha256": expected["sha256"]}
         and value.get("behavioral_validation") == "not_run", "v2_inspection_identity")
    return row

def input_identity(relative):
    if relative.startswith("models/"):
        return next(value for value in tracked_files.values() if value["relativePath"] == relative)
    need(relative in known_files and known_files[relative]["file"] is not None, "helper_unknown_file")
    return known_files[relative]["file"]

def metadata_signature(row):
    return row["kind"], row["arrayKind"], row["count"], row["encodedBytes"], row["encodedSha256"]

def expected_pair(reference, quantized):
    a = {row["name"]: row for row in reference["tensors"]}
    b = {row["name"]: row for row in quantized["tensors"]}
    need(len(a) == len(b) == 57 and set(a) == set(b)
         and all(a[n]["dimensions"] == b[n]["dimensions"] for n in a), "pair_inventory_dimensions")
    need(all(row["ggmlType"] == 0 for row in a.values())
         and dict(collections.Counter(row["ggmlType"] for row in b.values())) == {0: 13, 2: 43, 8: 1},
         "public_fixture_type_counts")
    need(reference["allowlistedScalars"].get("general.architecture") ==
         quantized["allowlistedScalars"].get("general.architecture") == {"kind": 8, "value": "llama"}, "pair_architecture")
    am = {row["key"]: metadata_signature(row) for row in reference["metadata"] if row["key"] not in QUANTIZATION_KEYS}
    bm = {row["key"]: metadata_signature(row) for row in quantized["metadata"] if row["key"] not in QUANTIZATION_KEYS}
    need(am == bm, "pair_metadata_hashes")
    unquantized = []
    for name, tensor in b.items():
        if tensor["ggmlType"] in (0, 1):
            original = a[name]
            need((tensor["ggmlType"], tensor["payloadBytes"], tensor["payloadSha256"]) ==
                 (original["ggmlType"], original["payloadBytes"], original["payloadSha256"]), "unquantized_fresh_payload_mismatch")
            unquantized.append({"name": name, "ggmlType": tensor["ggmlType"], "bytes": tensor["payloadBytes"],
                                "referenceSha256": original["payloadSha256"], "quantizedSha256": tensor["payloadSha256"]})
    need(len(unquantized) == 13, "thirteen_unquantized_bindings_required")
    selected = {"name": SELECTED_TENSOR, "dimensions": a[SELECTED_TENSOR]["dimensions"],
                "elements": a[SELECTED_TENSOR]["elements"], "reference_kind": a[SELECTED_TENSOR]["ggmlType"],
                "quantized_kind": b[SELECTED_TENSOR]["ggmlType"],
                "reference_sha256": a[SELECTED_TENSOR]["payloadSha256"],
                "quantized_sha256": b[SELECTED_TENSOR]["payloadSha256"],
                "reference_bytes": a[SELECTED_TENSOR]["payloadBytes"], "quantized_bytes": b[SELECTED_TENSOR]["payloadBytes"]}
    need(selected["dimensions"] == [288, 288] and selected["elements"] == MAX_ELEMENTS
         and selected["reference_kind"] == 0 and selected["quantized_kind"] == 2
         and selected["reference_bytes"] == 331776 and selected["quantized_bytes"] == 46656, "selected_public_matrix_contract")
    q8 = b["output.weight"]
    need(q8["ggmlType"] == 8 and q8["dimensions"] == [288, 32000] and q8["payloadBytes"] == 9792000, "opaque_q8_span_contract")
    return {"matched": True, "tensorCount": 57, "metadataOutsideFourQuantizationKeysMatched": True,
            "unquantizedCount": 13, "unquantizedBindings": unquantized, "selected": selected,
            "opaqueQ8": {"name": "output.weight", "dimensions": q8["dimensions"], "ggmlType": 8,
                         "bytes": q8["payloadBytes"], "sha256": q8["payloadSha256"]},
            "selectedFiniteValues": "not_evaluated", "lineage": "unverified", "behavioralValidation": "not_run"}

def binding_phase(phase):
    out, err, command = run("independent_bindings_" + phase,
        [sys.executable, "-I", "-B", "-S", str(copied_helper), "--phase", phase], 100,
        allowed=(0, 2), cwd=root / "output")
    prefix = b"QUANTOPSY_PUBLIC_GGUF_V2_BINDINGS_RESULT="
    need(out.startswith(prefix) and out.endswith(b"\n") and out.count(prefix) == 1, "bindings_original_marker")
    value = strict_json(out[len(prefix):-1])
    report["bindings"].append(value)
    need(command["exitCode"] == 0 and err == b"" and value.get("schemaVersion") == 1
         and value.get("sourceCommit") == source_commit and value.get("phase") == phase
         and value.get("status") == "verified" and value.get("productAccepted") is False
         and value.get("errors") == [], "independent_bindings_refused")
    checks = value["checks"]
    need(all(checks.get(k) is True for k in ("rootIdentityVerified", "allOpenedFilesUnchanged",
         "allDescriptorsClosed", "pairBindingsVerified"))
         and checks.get("candidateBindingsVerified") is (True if phase == "candidate" else None), "bindings_closure_checks")
    expected_paths = {"helpers/public_gguf_inventory.py", "helpers/public_gguf_v2_bindings.py",
                      "models/stories15M.gguf", "models/stories15M-q4_0.gguf"}
    if phase == "candidate":
        expected_paths |= {"output/repair-plan-v2.json", "output/candidate/plan.json",
                           "output/candidate/repair.json", "output/candidate/candidate.gguf"}
    rows = value["files"]
    need(len(rows) == len(expected_paths) and {r["relativePath"] for r in rows} == expected_paths, "bindings_opened_file_set")
    for row in rows:
        original = input_identity(row["relativePath"])
        identity_fields = {k: original[k] for k in ("device", "inode", "uid", "mode", "bytes", "mtimeNs", "ctimeNs")}
        need(row["before"] == row["after"] == identity_fields and row["sha256Before"] == row["sha256After"] == original["sha256"]
             and row["unchanged"] is True and row["closed"] is True, "bindings_opened_file_identity")
    need(len(value["inputs"]) == 2, "bindings_input_count")
    validation = {"phase": phase, "inputs": [], "candidate": None}
    report.setdefault("bindingValidations", []).append(validation)
    for fixture, row in zip(FIXTURES, value["inputs"]):
        need(row["fixture"] == fixture["id"] and row["file"] in rows
             and row["file"]["relativePath"] == "models/" + fixture["name"], "bindings_input_target")
        validation["inputs"].append(validate_inventory(root / "models" / fixture["name"], fixture["bytes"], row["inventory"]))
    pair = expected_pair(*(r["inventory"] for r in value["inputs"]))
    need(value["pair"] == pair, "pair_receipt_reconstruction")
    if phase == "pair":
        need(value["candidate"] is None, "pair_candidate_must_be_absent")
    else:
        item = value["candidate"]
        candidate_file = input_identity("output/candidate/candidate.gguf")
        need(item["file"] in rows and item["file"]["relativePath"] == candidate_file["relativePath"], "candidate_helper_file_target")
        validation["candidate"] = validate_inventory(root / candidate_file["relativePath"], candidate_file["bytes"], item["inventory"])
    verify_generated("after_bindings_" + phase)
    verify_inputs("after_bindings_" + phase)
    return value

def preflight():
    global root, identity, marker_identity, nonce, env, baseline, work_end, copied_helper
    need(sys.version_info >= (3, 11) and sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site
         and sys.dont_write_bytecode, "isolated_preinstalled_python_required")
    need(source_commit is not None and len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit), "source_commit_required")
    need(os.environ.get("CI") == "true" and os.environ.get("GITHUB_ACTIONS") == "true"
         and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY and os.environ.get("GITHUB_REF") == BRANCH
         and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
         and os.environ.get("SOURCE_COMMIT") == source_commit
         and os.environ.get("QUANTOPSY_ARTIFACT_PUBLIC_REPOSITORY") == "true", "approved_public_ci_required")
    event = strict_json(read(os.environ["GITHUB_EVENT_PATH"], 262144))
    need(("ref" not in event or event["ref"] in (BRANCH, BRANCH.removeprefix("refs/heads/")))
         and event.get("repository", {}).get("private") is False
         and event.get("repository", {}).get("full_name") == REPOSITORY, "approved_dispatch_event_required")
    need(not any(os.environ.get(name) for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE", "PYTHONHTTPSVERIFY",
                                                  "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")), "custom_tls_environment_refused")
    now, monotonic = time.time(), time.monotonic()
    started = os.environ.get("QUANTOPSY_ARTIFACT_JOB_STARTED_MS", "")
    need(len(started) == 13 and started.isascii() and started.isdigit(), "job_clock_required")
    started = int(started) / 1000
    need(started <= now and now - started < 900, "job_clock_invalid")
    allowed = min(600, started + 900 - 90 - 30 - now)
    need(allowed >= 120, "job_entry_window_unavailable")
    work_end = monotonic + allowed
    report["timing"] = {"earliestStepAtUtc": datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
                        "entryAtUtc": utc(), "setupElapsedMs": round((now - started) * 1000),
                        "admittedWorkMs": round(allowed * 1000), "cleanupReservedMs": 90000, "marginReservedMs": 30000}
    original_home = Path(os.environ.get("HOME", ""))
    need(original_home.is_absolute() and original_home.is_dir(), "original_home_required")
    parent = Path(os.environ.get("RUNNER_TEMP", "")).resolve(strict=True)
    need(parent.is_absolute() and parent.is_dir() and repo != parent and repo not in parent.parents, "runner_temp_required")
    root = Path(tempfile.mkdtemp(prefix="qv-", dir=parent))
    root.chmod(0o700)
    info = root.lstat()
    identity = (info.st_dev, info.st_ino)
    nonce = secrets.token_hex(16)
    fd = os.open(root / ".owner", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(nonce.encode())
    marker = (root / ".owner").lstat()
    marker_identity = (marker.st_dev, marker.st_ino)
    for name in ("models", "tmp", "output", "config", "cache", "data", "state", "helpers"):
        (root / name).mkdir(mode=0o700)
        known_directories[name] = file_identity((root / name).lstat())
    owner()
    python = Path(sys.executable).resolve(strict=True)
    report["python"] = {"version": sys.version, "executableBasename": python.name, "bytes": python.stat().st_size,
                        "sha256": sha(read(python, 268435456)), "isolatedController": True,
                        "applicationUsesNoSiteAndNoBytecode": True, "installedByPacket": False}
    env = {"PATH": os.pathsep.join([str(python.parent), "/usr/local/bin", "/usr/bin", "/bin"]),
           "HOME": str(original_home), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
           "CI": "true", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "TMPDIR": str(root / "tmp"), "TMP": str(root / "tmp"), "TEMP": str(root / "tmp"),
           "XDG_CONFIG_HOME": str(root / "config"), "XDG_CACHE_HOME": str(root / "cache"),
           "XDG_DATA_HOME": str(root / "data"), "XDG_STATE_HOME": str(root / "state"),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "SOURCE_COMMIT": source_commit,
           "QUANTOPSY_ARTIFACT_RUN_ROOT": str(root), "QUANTOPSY_ARTIFACT_NONCE": nonce,
           "QUANTOPSY_ARTIFACT_BINDINGS_SHA256": HELPER_SHA256}
    for folder in Path("/proc").iterdir():
        if folder.name.isdigit():
            try:
                item = proc_info(int(folder.name))
                baseline[item["pid"]] = item["startTime"]
            except (FileNotFoundError, ProcessLookupError):
                pass
    storage_sample("initial")
    canonical_base()
    guard("before")
    for source, target, expected in (
            (INVENTORY_HELPER, "helpers/public_gguf_inventory.py", INVENTORY_SHA256),
            (HELPER, "helpers/public_gguf_v2_bindings.py", HELPER_SHA256)):
        data = read(repo / source, 128 * 1024)
        need(sha(data) == expected, "independent_helper_source_changed")
        own = owned_write(target, data)
        need(own["sha256"] == expected, "independent_helper_copy_changed")
    copied_helper = root / "helpers/public_gguf_v2_bindings.py"
    report["homePreserved"] = env["HOME"] == str(original_home)


def read_owned_json(relative):
    need(relative in known_files and known_files[relative]["file"] is not None, "unregistered_json_output")
    before = known_files[relative]["file"]
    need(before["bytes"] <= 2 * 1024**2, "json_output_bound")
    raw = read(root / relative, 2 * 1024**2)
    need(fingerprint(root / relative) == before and sha(raw) == before["sha256"], "json_output_changed")
    original = capture(raw, len(raw), sha(raw), True)
    report.setdefault("jsonFileOriginals", []).append({"relativePath": relative, "original": original})
    need(original["retained"] and original["complete"], "json_original_incomplete")
    value = strict_json(raw)
    need(canonical(value) == raw, "product_json_not_canonical")
    return value, raw

def validate_plan(pair):
    value, raw = read_owned_json("output/repair-plan-v2.json")
    expected = {"kind": "quantopsy.repair-plan", "schema_version": 2,
                "reference": {"name": FIXTURES[0]["name"], "size": FIXTURES[0]["bytes"], "sha256": FIXTURES[0]["sha256"]},
                "quantized": {"name": FIXTURES[1]["name"], "size": FIXTURES[1]["bytes"], "sha256": FIXTURES[1]["sha256"]},
                "architecture": "llama", "provenance": PROVENANCE, "selected": [pair["selected"]],
                "added_weight_bytes": MAX_ADDED_BYTES, "max_added_bytes": MAX_ADDED_BYTES, "max_elements": MAX_ELEMENTS,
                "lineage": "operator_declared_compatibility_checked", "behavioral_validation": "not_run"}
    need(value == expected and type(value["schema_version"]) is int and all(type(value[k]) is int for k in
         ("added_weight_bytes", "max_added_bytes", "max_elements")), "plan_does_not_match_authored_intent")
    need(read(root / "output/provenance.json") == canonical(PROVENANCE), "authored_provenance_changed")
    report["plan"] = {"value": value, "bytes": len(raw), "sha256": sha(raw), "exactRequestedIntent": True}
    return value, raw

def validate_candidate(pair_receipt, candidate_receipt, plan, plan_raw, cli_manifest, cli_raw):
    manifest, manifest_raw = read_owned_json("output/candidate/repair.json")
    bundled, bundled_raw = read_owned_json("output/candidate/plan.json")
    need(bundled_raw == plan_raw and bundled == plan and cli_raw == manifest_raw and cli_manifest == manifest,
         "plan_manifest_originals_differ")
    candidate = input_identity("output/candidate/candidate.gguf")
    expected = {key: value for key, value in plan.items() if key not in ("schema_version", "kind")}
    quantized_inventory = candidate_receipt["inputs"][1]["inventory"]
    expected.update({"schema_version": 2, "kind": "quantopsy.repair-result", "plan_sha256": sha(plan_raw),
                     "output": {"name": "candidate.gguf", "size": candidate["bytes"], "sha256": candidate["sha256"]},
                     "reserve_bytes": RESERVE_BYTES,
                     "metadata_omissions": ["general.file_type"] if any(r["key"] == "general.file_type" for r in quantized_inventory["metadata"]) else [],
                     "verification": {"container_reparsed": True, "metadata_preserved": True,
                                      "tensor_payloads_verified": True, "source_hashes_verified": True}})
    need(manifest == expected and type(manifest["reserve_bytes"]) is int
         and all(type(v) is bool and v is True for v in manifest["verification"].values()),
         "manifest_does_not_match_authored_intent")
    need(candidate_receipt["pair"] == pair_receipt["pair"], "pair_changed_during_plan_repair")
    reference = {r["name"]: r for r in candidate_receipt["inputs"][0]["inventory"]["tensors"]}
    quantized = {r["name"]: r for r in quantized_inventory["tensors"]}
    current = candidate_receipt["candidate"]["inventory"]
    need([r["name"] for r in current["tensors"]] == [r["name"] for r in quantized_inventory["tensors"]]
         and current["tensorData"]["alignment"] == quantized_inventory["tensorData"]["alignment"], "candidate_tensor_order_or_alignment")
    q_metadata = [r for r in quantized_inventory["metadata"] if r["key"] != "general.file_type"]
    need([r["key"] for r in current["metadata"]] == [r["key"] for r in q_metadata]
         and all(metadata_signature(a) == metadata_signature(b) for a, b in zip(current["metadata"], q_metadata)),
         "candidate_encoded_metadata_changed")
    payloads = []
    for actual in current["tensors"]:
        name = actual["name"]
        source = reference[name] if name == SELECTED_TENSOR else quantized[name]
        need((actual["dimensions"], actual["ggmlType"], actual["payloadBytes"], actual["payloadSha256"]) ==
             (source["dimensions"], source["ggmlType"], source["payloadBytes"], source["payloadSha256"]),
             "candidate_prescribed_payload_mismatch")
        payloads.append({"name": name, "source": "reference" if name == SELECTED_TENSOR else "quantized",
                         "ggmlType": actual["ggmlType"], "bytes": actual["payloadBytes"],
                         "expectedSha256": source["payloadSha256"], "candidateSha256": actual["payloadSha256"]})
    need(len(payloads) == 57 and sum(r["source"] == "reference" for r in payloads) == 1, "candidate_payload_count")
    bindings = {"matched": True, "tensorCount": 57, "selectedCount": 1, "unselectedCount": 56,
                "payloadBindings": payloads, "planSha256": sha(plan_raw), "manifestSha256": sha(manifest_raw),
                "candidateSha256": candidate["sha256"], "candidateBytes": candidate["bytes"],
                "declaredProvenanceSha256": sha(canonical(PROVENANCE)),
                "metadataOrderAndBytesPreservedMinusFileType": True, "behavioralValidation": "not_run"}
    need(candidate_receipt["candidate"]["bindings"] == bindings, "candidate_receipt_reconstruction")
    report["candidate"] = {"value": manifest, "manifestBytes": len(manifest_raw), "manifestSha256": sha(manifest_raw),
                            "bindings": bindings, "exactRequestedIntent": True, "behavioralValidation": "not_run"}

def cleanup():
    global cleanup_end
    cleanup_end = min(time.monotonic() + 90, (work_end + 90) if work_end is not None else time.monotonic() + 90)
    result = {"passed": False, "signals": [], "failures": [], "remaining": [], "uncertain": []}
    report["cleanup"] = result
    if root is None or identity is None:
        result["rootCreated"] = False
        return
    try:
        guard("before_cleanup", cleanup=True)
    except Exception as error:
        result["failures"].append("source_" + finite(error))
    try:
        owner()
        for process in processes:
            if process.poll() is None:
                if process.pid not in groups:
                    result["failures"].append("unattributed_child_remaining")
                    continue
                if signal_original_group(process.pid, signal.SIGTERM):
                    result["signals"].append({"pid": process.pid, "signal": "SIGTERM"})
        end = min(time.monotonic() + 3, cleanup_end - 10)
        while time.monotonic() < end and any(p.poll() is None for p in processes):
            time.sleep(0.05)
        for process in processes:
            if process.poll() is None and process.pid in groups:
                if signal_original_group(process.pid, signal.SIGKILL):
                    result["signals"].append({"pid": process.pid, "signal": "SIGKILL"})
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.01, min(1, cleanup_end - time.monotonic())))
                except subprocess.TimeoutExpired:
                    result["failures"].append("child_unreaped")
        rows, uncertain = owned()
        result.update({"remaining": rows, "uncertain": uncertain, "attributedPidStarttimeObservations": list(attributed.values()),
                       "allOriginalGroupsAbsent": all(group_gone(pid) for pid in groups),
                       "allChildrenReaped": all(p.poll() is not None for p in processes)})
        need(not rows and not uncertain and result["allOriginalGroupsAbsent"] and result["allChildrenReaped"], "process_cleanup_unverified")
        # A failed writer can leave known output names. Admit only these now-closed
        # paths, and leave unknown/partial/replaced files in place with a failure.
        collect_outputs()
        verify_inputs("before_removal")
        verify_generated("before_removal")
        allowed = {".owner": {"bytes": 32, "sha256": sha(nonce.encode()),
                              "createdDevice": str(marker_identity[0]), "createdInode": str(marker_identity[1])}}
        for relative, row in known_files.items():
            need(row["file"] is not None, "generated_cleanup_identity_missing")
            allowed[relative] = {"bytes": row["file"]["bytes"], "sha256": row["file"]["sha256"],
                                 "createdDevice": row["createdDevice"], "createdInode": row["createdInode"]}
        for fixture in FIXTURES:
            if fixture["id"] in tracked_files:
                current = tracked_files[fixture["id"]]
                allowed[current["relativePath"]] = {"bytes": current["bytes"], "sha256": current["sha256"],
                                                    "createdDevice": current["device"], "createdInode": current["inode"]}
        for download_row in report["downloads"]:
            partial = download_row.get("partialFile")
            if partial is not None:
                allowed[partial["relativePath"]] = {"bytes": partial["writtenBytes"], "sha256": partial["writtenSha256"],
                                                   "createdDevice": partial["createdDevice"], "createdInode": partial["createdInode"]}
        checks, actual, directories_seen = [], set(), set()
        for parent, directories, names in os.walk(root, followlinks=False):
            for name in directories:
                directory = Path(parent) / name
                relative = str(directory.relative_to(root))
                need(relative in known_directories, "unexpected_directory_preserved")
                current = file_identity(directory.lstat())
                expected = known_directories[relative]
                need(not directory.is_symlink() and directory.is_dir() and current["uid"] == os.getuid()
                     and current["mode"] == "0700" and current["device"] == expected["device"]
                     and current["inode"] == expected["inode"], "owned_directory_replaced")
                directories_seen.add(relative)
            for name in names:
                file = Path(parent) / name
                relative = str(file.relative_to(root))
                need(relative in allowed, "unexpected_file_preserved")
                expected = allowed[relative]
                current = fingerprint(file, expected)
                need(current["device"] == expected["createdDevice"] and current["inode"] == expected["createdInode"],
                     "owned_path_replaced")
                if relative in known_files:
                    need(current == known_files[relative]["file"], "generated_changed_before_removal")
                if relative.startswith("models/") and not relative.endswith(".partial"):
                    need(current == next(v for v in tracked_files.values() if v["relativePath"] == relative), "model_changed_before_removal")
                checks.append(current)
                actual.add(relative)
        need(actual == set(allowed) and directories_seen == set(known_directories), "owned_entry_set_changed")
        storage_sample("before_removal")
        # Recheck every recorded file and directory identity immediately before
        # deletion; unexpected fixture residue is never silently removed.
        for previous in checks:
            need(fingerprint(root / previous["relativePath"]) == previous, "owned_file_changed_at_removal")
        for relative, previous in known_directories.items():
            current = file_identity((root / relative).lstat())
            need(current["device"] == previous["device"] and current["inode"] == previous["inode"]
                 and current["uid"] == os.getuid() and current["mode"] == "0700", "owned_directory_changed_at_removal")
        owner()
        result["ownedFileChecks"] = checks
        result["ownedDirectories"] = sorted(directories_seen)
        shutil.rmtree(root)
        need(not root.exists(), "scratch_still_present")
        report["scratchCleanup"] = "passed"
        guard("after_cleanup", cleanup=True, post_scratch=True)
        rows, uncertain = owned()
        result.update({"remaining": rows, "uncertain": uncertain,
                       "allOriginalGroupsAbsent": all(group_gone(pid) for pid in groups),
                       "allChildrenReaped": all(p.poll() is not None for p in processes)})
        need(not rows and not uncertain and result["allOriginalGroupsAbsent"] and result["allChildrenReaped"],
             "post_cleanup_process_remaining")
        result["passed"] = not result["failures"] and not result["signals"]
    except Exception as error:
        result["failures"].append(finite(error))
        report.setdefault("scratchCleanup", "unverified")

def main():
    completed = False
    def stop(_signal, _frame):
        global interrupted
        interrupted = True
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop)
    try:
        preflight()
        fixture_checks()
        need(not any((root / "tmp").iterdir()), "fixture_temporary_residue_preserved")
        verify_generated("after_fixtures")
        for fixture in FIXTURES:
            download(fixture)
        need(len(report["downloads"]) == 2 and sum(r["receivedBytes"] for r in report["downloads"]) == DOWNLOAD_TOTAL
             and all(r["complete"] and r["bodyEOFObserved"] and r["responseClosed"] and r["fileDescriptorClosed"]
                     and not r["resourceErrors"] for r in report["downloads"]), "downloads_incomplete")
        verify_inputs("after_downloads")
        guard("after_downloads")
        inspections = [inspect_file("v2_inspect_reference" if n == 0 else "v2_inspect_quantized",
                                   root / "models" / fixture["name"], fixture) for n, fixture in enumerate(FIXTURES)]
        pair_receipt = binding_phase("pair")
        for row, fixture, independent in zip(inspections, FIXTURES, pair_receipt["inputs"]):
            validate_inspection(fixture, independent["inventory"], row["value"])
            row["boundToIndependentInventory"] = True
        report["pairCompatibility"] = {"accepted": True, "independent": pair_receipt["pair"], "profile": "v2",
                                       "lineage": "unverified", "behavioralValidation": "not_run"}
        owned_write("output/provenance.json", canonical(PROVENANCE))
        report["authoredProvenance"] = {"value": PROVENANCE, "bytes": len(canonical(PROVENANCE)),
                                        "sha256": sha(canonical(PROVENANCE)), "lineage": "unverified"}
        need(not os.path.lexists(root / "output/repair-plan-v2.json"), "external_plan_not_absent")
        plan_status, _ = product_command("v2_plan_explicit_matrix",
            ["plan", root / "models" / FIXTURES[0]["name"], root / "models" / FIXTURES[1]["name"], "--profile", "v2",
             "--tensor", SELECTED_TENSOR, "--provenance", root / "output/provenance.json",
             "--max-added-bytes", MAX_ADDED_BYTES, "--max-elements", MAX_ELEMENTS,
             "--output", root / "output/repair-plan-v2.json"], outputs=True)
        need(plan_status == {"saved_file": "repair-plan-v2.json", "status": "created"}, "plan_create_ack")
        plan, plan_raw = validate_plan(pair_receipt["pair"])
        verify_inputs("after_plan")
        verify_generated("after_plan")
        need(not os.path.lexists(root / "output/candidate"), "candidate_directory_not_absent")
        cli_manifest, cli_raw = product_command("v2_repair_explicit_plan",
            ["repair", root / "models" / FIXTURES[0]["name"], root / "models" / FIXTURES[1]["name"], "--profile", "v2",
             "--plan", root / "output/repair-plan-v2.json", "--output-dir", root / "output/candidate",
             "--reserve-bytes", RESERVE_BYTES], outputs=True)
        candidate = input_identity("output/candidate/candidate.gguf")
        need(candidate["bytes"] <= 32 * 1024**2, "candidate_file_bound")
        candidate_expected = {"name": "candidate.gguf", "bytes": candidate["bytes"], "sha256": candidate["sha256"]}
        final_inspection = inspect_file("v2_inspect_candidate", root / candidate["relativePath"], candidate_expected)
        candidate_receipt = binding_phase("candidate")
        validate_candidate(pair_receipt, candidate_receipt, plan, plan_raw, cli_manifest, cli_raw)
        validate_inspection(candidate_expected, candidate_receipt["candidate"]["inventory"], final_inspection["value"])
        final_inspection["boundToIndependentInventory"] = True
        verify_inputs("after_candidate_verification")
        verify_generated("after_candidate_verification")
        guard("after_observations")
        completed = True
    except Exception as error:
        report["errors"].append(finite(error))
        frame = error.__traceback__
        while frame is not None and frame.tb_next is not None:
            frame = frame.tb_next
        report["errorLocation"] = None if frame is None else {"file": Path(frame.tb_frame.f_code.co_filename).name[:100], "line": frame.tb_lineno}
        report["errorType"] = type(error).__name__[:80]
    finally:
        cleanup()
        if interrupted:
            report["errors"].append("interrupted")
        lifecycle = all(row.get("normallyClosed") is True for row in report["commands"])
        zero = all(row["exitCode"] == 0 for row in report["commands"])
        report["commandOutcomes"] = [{"label": row["label"], "exitCode": row["exitCode"],
                                      "classification": "zero_exit" if row["exitCode"] == 0 else "unexpected_exit"}
                                     for row in report["commands"]]
        report["processCleanup"] = {"allRowsNormallyClosed": lifecycle, "allCommandExitsZero": zero,
                                    "originalCommands": len(report["commands"]),
                                    "signalsSent": any(row["termSent"] or row["killSent"] for row in report["commands"])}
        report["compatibilityAndCandidateVerified"] = bool(completed and not report["errors"] and not interrupted
            and lifecycle and zero and report.get("cleanup", {}).get("passed") is True
            and report.get("scratchCleanup") == "passed" and len(report["guards"]) == 5
            and all(g["passed"] for g in report["guards"]) and report.get("fixture", {}).get("accepted") is True
            and work_end is not None and time.monotonic() <= work_end + 90)
        report["packetStatus"] = "verified" if report["compatibilityAndCandidateVerified"] else "failed"
        report["completedAtUtc"] = utc()
        output = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        if len(output) > REPORT_LIMIT:
            report["packetStatus"], report["compatibilityAndCandidateVerified"] = "failed", False
            report["errors"].append("report_bound_parsed_duplicates_omitted")
            report["bindings"] = [{"omittedParsedCopy": True, "originalRetainedInCommand": "independent_bindings_" + r["phase"]}
                                  for r in report["bindings"]]
            for row in report.get("applicationCommands", []):
                row.pop("stdoutJson", None)
                row.pop("stderrJson", None)
            output = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        need(len(output) <= REPORT_LIMIT, "report_bound")
        signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
        signal.setitimer(signal.ITIMER_REAL, 2)
        sys.stdout.buffer.write(b"QUANTOPSY_PUBLIC_GGUF_V2_COMPATIBILITY_RESULT=" + output + b"\n")
        sys.stdout.buffer.flush()
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if report["compatibilityAndCandidateVerified"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
