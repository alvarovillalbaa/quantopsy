#!/usr/bin/env python3
"""One bounded, predeclared four-role input-embedding study; public CI only."""

import ast
import datetime
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import resource
import secrets
import shutil
import signal
import socket
import ssl
import stat
import sys
import tarfile
import tempfile
import time
import types
import urllib.request


BASE_COMMIT = "b60d04db5d45fbd317dea28add2a2cb609cc4c97"
REPOSITORY = "alvarovillalbaa/quantopsy"
BRANCH = "refs/heads/qa/quantopsy-embedding-20261113-1"
ACTIVE = ".github/workflows/embedding-study-v2.yml"
CONTROLLER = "tests/scripts/run-embedding-study-v2.py"
SUPPORT = "tests/scripts/run-public-gguf-v2-compatibility.py"
SUPPORT_SHA256 = "5161f1b9f89f6b8534f2e036dce06850082e4579e427417fa1ba0411cce9b99d"
DATA_HELPER = "tests/helpers/embedding_study_v2.py"
SCORER = "tests/helpers/behavioral_scorer.cpp"
CMAKE = "tests/helpers/behavioral_engine/CMakeLists.txt"
LLAMA_REVISION = "74d4f5b041ad837153b0e90fc864b8290e01d8d5"
LLAMA_TREE = "9ae6f9dd0984b780c942ccf510e9cd5fcee9e724"
ARCHIVE_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/" + LLAMA_REVISION
ARCHIVE_ROOT = "llama.cpp-" + LLAMA_REVISION
MODIFIED_PRIOR = {"AGENTS.md"}
PRIOR_RECEIPT = "docs/results/2026/11-13/profile-v3-checks.json"
PRIOR_RECEIPT_SHA256 = "143f44e45d017c7f0f686922cd8608253e16d96e55c6c874e23a7842835c0403"
ADDED_PATHS = {CONTROLLER, DATA_HELPER, "docs/proposals/embedding-study-v2-execution.md",
               "docs/proposals/embedding-study-v2.workflow.yml.example"}
PROTOCOL = "docs/studies/embedding-v2.json"
PROTOCOL_SHA256 = "8c35578fb0f8a2451a876f0a86cacf98941f7887a289df0bfd19c5ae416c5bc3"
MAX_ARCHIVE = 64 * 1024**2
MAX_EXTRACTED = 512 * 1024**2
MAX_BUILD = 2 * 1024**3
MAX_STORAGE = 3 * 1024**3
MAX_FILES = 30_000
MAX_DIRECTORIES = 8_000
MAX_ELEMENTS = 15_187_968
ADDED_WEIGHT_BYTES = 52_208_640
MAX_ADDED_BYTES = 64 * 1024**2
RESERVE_BYTES = 1024**3
METRICS = [
    {"id": "mean_target_nll", "direction": "lower", "min_quantization_loss": "0.005",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.02", "max_quantized_regression": "0.01"},
    {"id": "target_token_accuracy", "direction": "higher", "min_quantization_loss": "0.02",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.03", "max_quantized_regression": "0.02"},
]
PREFIX = b"QUANTOPSY_EMBEDDING_STUDY_V2_RESULT="
_c = None
_source_tree = None
_data = None
_frozen = {}
_binary = None
_archive_owner = None
_tree_frozen = set()


def need(value, code):
    if not value:
        raise RuntimeError(code)


def require_entry(value, code, path, info):
    if not value:
        relative = str(Path(path).relative_to(_c.root))
        diagnostics = _c.report.setdefault("entryRefusals", [])
        if len(diagnostics) < 16:
            diagnostics.append({"code": code, "relativePath": relative[:1024],
                                "mode": format(stat.S_IMODE(info.st_mode), "04o"), "bytes": info.st_size,
                                "type": "file" if stat.S_ISREG(info.st_mode) else "directory" if stat.S_ISDIR(info.st_mode) else "other",
                                "uidMatchesOwner": info.st_uid == os.getuid()})
        need(False, code)


def private_build_descendant(path):
    relative = Path(path).relative_to(_c.root)
    if len(relative.parts) < 2 or relative.parts[0] != "build":
        return False
    private = _c.root / "build"
    value = private.lstat()
    original = _c.known_directories.get("build")
    require_entry(original is not None and stat.S_ISDIR(value.st_mode) and not private.is_symlink()
                  and value.st_uid == os.getuid() and stat.S_IMODE(value.st_mode) == 0o700
                  and (str(value.st_dev), str(value.st_ino)) == (original["device"], original["inode"]),
                  "private_build_ancestor_changed", private, value)
    parent = Path(path).parent
    while parent != private:
        value = parent.lstat()
        require_entry(stat.S_ISDIR(value.st_mode) and not parent.is_symlink() and value.st_uid == os.getuid()
                      and stat.S_IMODE(value.st_mode) in (0o700, 0o755), "build_ancestor_type_or_mode", parent, value)
        parent = parent.parent
    return True


def load_fixed(path, name, expected=None):
    raw = path.read_bytes()
    need(len(raw) <= 128 * 1024 and (expected is None or hashlib.sha256(raw).hexdigest() == expected), "fixed_support_source_changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Execute the already hashed bytes, rather than reading a path again.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def storage_sample(label):
    _c.owner()
    total = files = directories = extracted = built = 0
    for parent, folders, names in os.walk(_c.root, followlinks=False):
        directories += len(folders)
        for name in folders + names:
            path = Path(parent) / name
            value = path.lstat()
            need(not path.is_symlink() and value.st_uid == os.getuid()
                 and (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)), "unexpected_scratch_entry")
            if stat.S_ISREG(value.st_mode):
                files += 1
                total += value.st_size
                relative = path.relative_to(_c.root).parts
                extracted += value.st_size if relative[0] == "upstream" else 0
                built += value.st_size if relative[0] == "build" else 0
    _c.report["storageSamples"].append({"label": label, "logicalBytes": total, "files": files,
                                      "directories": directories, "extractedBytes": extracted, "buildBytes": built})
    need(total <= MAX_STORAGE and extracted <= MAX_EXTRACTED and built <= MAX_BUILD
         and files <= MAX_FILES and directories <= MAX_DIRECTORIES, "scratch_storage_bound")
    free = shutil.disk_usage(_c.root).free
    _c.report["storageSamples"][-1]["freeBytes"] = free
    need(free >= 2 * 1024**3, "runner_free_space_reserve")


def fingerprint(file, expected=None):
    _c.owner()
    file = Path(file)
    need(_c.root in file.parents and not file.is_symlink(), "input_outside_owned_root")
    for parent in file.parents:
        if parent == _c.root:
            break
        need(parent.is_dir() and not parent.is_symlink(), "input_ancestor_changed")
    before = file.lstat()
    modes = (0o600, 0o644, 0o700, 0o755) if private_build_descendant(file) else (0o600, 0o700)
    require_entry(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                  and stat.S_IMODE(before.st_mode) in modes and before.st_size <= MAX_BUILD,
                  "input_type_or_bound", file, before)
    fd = os.open(file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest = hashlib.sha256()
    try:
        need(_c.file_identity(os.fstat(fd)) == _c.file_identity(before), "input_changed_at_open")
        remaining = before.st_size
        while remaining:
            need(time.monotonic() < (_c.cleanup_end if _c.cleanup_end is not None else _c.work_end), "hash_deadline")
            chunk = os.read(fd, min(1024**2, remaining))
            need(bool(chunk), "input_truncated")
            digest.update(chunk)
            remaining -= len(chunk)
        need(os.read(fd, 1) == b"" and _c.file_identity(before) == _c.file_identity(os.fstat(fd))
             == _c.file_identity(file.lstat()), "input_changed_during_hash")
    finally:
        os.close(fd)
    result = {"relativePath": str(file.relative_to(_c.root)), **_c.file_identity(before), "sha256": digest.hexdigest()}
    if expected is not None:
        need(result["bytes"] == expected["bytes"] and result["sha256"] == expected["sha256"], "input_identity_mismatch")
    return result


def admit_file(relative, *, mutable=False):
    current = fingerprint(_c.root / relative)
    if relative in _c.known_files:
        old = _c.known_files[relative]
        need((current["device"], current["inode"]) == (old["createdDevice"], old["createdInode"]), "generated_file_replaced")
        if not mutable:
            need(current == old["file"], "generated_file_changed")
    _c.known_files[relative] = {"createdDevice": current["device"], "createdInode": current["inode"], "file": current}
    return current


def admit_directory(relative):
    path = _c.root / relative
    value = path.lstat()
    modes = (0o700, 0o755) if private_build_descendant(path) else (0o700,)
    require_entry(stat.S_ISDIR(value.st_mode) and not path.is_symlink() and value.st_uid == os.getuid()
                  and stat.S_IMODE(value.st_mode) in modes, "owned_directory_type", path, value)
    current = _c.file_identity(value)
    if relative in _c.known_directories:
        old = _c.known_directories[relative]
        require_entry((current["device"], current["inode"], current["mode"]) == (old["device"], old["inode"], old["mode"]),
                      "owned_directory_replaced_or_mode_changed", path, value)
    _c.known_directories[relative] = current


def admit_tree(relative, *, mutable=False):
    """Admit tool outputs only inside a pre-created, exclusively owned tree."""
    need(relative in ("build", "upstream"), "unexpected_generated_tree")
    first = relative not in _tree_frozen
    need(first or not mutable, "sealed_generated_tree_mutation_refused")
    old = {name for name in _c.known_files if name.startswith(relative + "/")}
    old_directories = {name for name in _c.known_directories if name == relative or name.startswith(relative + "/")}
    admit_directory(relative)
    rows = []
    directories_seen = {relative}
    for parent, directories, names in os.walk(_c.root / relative, followlinks=False):
        need(len(rows) <= MAX_FILES, "generated_tree_bound")
        for name in directories:
            name = str((Path(parent) / name).relative_to(_c.root))
            need(first or name in old_directories, "sealed_generated_tree_new_directory")
            admit_directory(name)
            directories_seen.add(name)
        for name in names:
            name = str((Path(parent) / name).relative_to(_c.root))
            need(first or name in old, "sealed_generated_tree_new_file")
            rows.append(admit_file(name, mutable=mutable))
    present = {row["relativePath"] for row in rows}
    if not first:
        need(old == present and old_directories == directories_seen, "generated_tree_entry_set_changed")
    else:
        need(old <= present and old_directories <= directories_seen, "generated_tree_entry_disappeared")
    _tree_frozen.add(relative)
    storage_sample("admit_" + relative)
    return {"files": len(rows), "bytes": sum(row["bytes"] for row in rows),
            "vectorSha256": _c.sha(_c.canonical(sorted(rows, key=lambda row: row["relativePath"])))}


def collect_outputs():
    _c.owner()
    for profile in ("v2", "v3"):
        directory = "output/candidate-" + profile
        if os.path.lexists(_c.root / directory):
            admit_directory(directory)
        for name in ("output/repair-plan-" + profile + ".json", directory + "/candidate.gguf",
                     directory + "/plan.json", directory + "/repair.json", "output/comparison-" + profile + ".json"):
            if os.path.lexists(_c.root / name):
                admit_file(name)


def verify_generated(label):
    rows = []
    for relative, expected in _c.known_files.items():
        need(expected["file"] is not None and fingerprint(_c.root / relative) == expected["file"], "generated_file_changed_between_phases")
        rows.append(expected["file"])
    _c.report.setdefault("generatedChecks", []).append({"label": label, "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows), "vectorSha256": _c.sha(_c.canonical(sorted(rows, key=lambda row: row["relativePath"])))})


def source_guard(label, cleanup=False, post_scratch=False):
    global _source_tree
    item = {"label": label, "passed": False}
    _c.report["guards"].append(item)
    options = {"cleanup": cleanup, "post_scratch": post_scratch}
    out, _, _ = _c.run(label + "_head", ["git", "rev-parse", "HEAD"], 10, **options)
    need(out.strip().decode() == _c.source_commit, "source_head_changed")
    need(not os.path.lexists(_c.repo / ACTIVE), "active_workflow_must_not_be_in_checked_source")
    raw, _, _ = _c.run(label + "_tree", ["git", "ls-tree", "-r", "-z", "HEAD"], 10, **options)
    index, _, _ = _c.run(label + "_index", ["git", "ls-files", "--stage", "-z"], 10, **options)
    tree, index = _c.parse_git(raw), _c.parse_git(index, index=True)
    need(tree == index and len(tree) <= 256, "source_index_or_count_changed")
    if _source_tree is None:
        parent, _, _ = _c.run("canonical_direct_parent", ["git", "rev-parse", "HEAD^"], 10)
        need(parent.strip().decode() == BASE_COMMIT, "canonical_direct_parent_changed")
        base, _, _ = _c.run("canonical_base_tree", ["git", "ls-tree", "-r", "-z", BASE_COMMIT], 10)
        base = _c.parse_git(base)
        need(set(base) <= set(tree) and set(tree) - set(base) == ADDED_PATHS, "source_scope_added_or_removed")
        need(all(tree[name] == record or name in MODIFIED_PRIOR for name, record in base.items()), "unmodified_prior_source_changed")
        old_agents, _, _ = _c.run("canonical_agents_prefix", ["git", "show", BASE_COMMIT + ":AGENTS.md"], 10)
        need(_c.read(_c.repo / "AGENTS.md").startswith(old_agents), "prior_agents_must_be_preserved_verbatim")
        _source_tree = tree
        _c.report["canonicalBaseline"] = {"commit": BASE_COMMIT, "blobs": len(base), "allowedPriorModifications": sorted(MODIFIED_PRIOR),
                                           "onlyAddedPaths": sorted(ADDED_PATHS)}
    need(tree == _source_tree, "source_tree_changed")
    vector = [_c.file_vector(name) for name in sorted(tree)]
    need(all(row["gitBlob"] == tree[row["path"]]["gitBlob"] and row["mode"] == tree[row["path"]]["mode"] for row in vector),
         "source_body_or_mode_changed")
    for suffix, args in (("status", ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]),
                         ("ignored", ["git", "status", "--porcelain=v1", "-z", "--ignored=matching", "--untracked-files=normal"])):
        out, _, _ = _c.run(label + "_" + suffix, args, 10, **options)
        need(out == b"", "source_generated_or_dirty_output")
    _c.run(label + "_diff", ["git", "-c", "core.filemode=true", "diff", "--no-ext-diff", "--no-textconv", "--quiet", "--exit-code", "HEAD", "--"], 10, **options)
    if _c.source_vector is None:
        _c.source_vector = vector
        _c.report["sourceFiles"] = vector
    need(vector == _c.source_vector, "source_vector_changed")
    item.update({"passed": True, "blobs": len(vector), "bytes": sum(row["bytes"] for row in vector),
                 "vectorSha256": _c.sha(_c.canonical(vector)), "fullGitIndexNativeBindings": True})


def preflight():
    need(sys.platform == "linux" and sys.version_info >= (3, 11) and sys.flags.isolated
         and sys.flags.no_site and sys.dont_write_bytecode, "isolated_preinstalled_python_required")
    need(_c.source_commit is not None and len(_c.source_commit) == 40
         and all(ch in "0123456789abcdef" for ch in _c.source_commit), "source_commit_required")
    need(os.environ.get("CI") == "true" and os.environ.get("GITHUB_ACTIONS") == "true"
         and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY and os.environ.get("GITHUB_REF") == BRANCH
         and os.environ.get("GITHUB_EVENT_NAME") == "create"
         and os.environ.get("SOURCE_COMMIT") == _c.source_commit
         and os.environ.get("QUANTOPSY_ARTIFACT_PUBLIC_REPOSITORY") == "true", "approved_public_ci_required")
    event = _c.strict_json(_c.read(os.environ["GITHUB_EVENT_PATH"], 262144))
    need(event.get("ref") == BRANCH.removeprefix("refs/heads/") and event.get("ref_type") == "branch"
         and event.get("repository", {}).get("private") is False
         and event.get("repository", {}).get("full_name") == REPOSITORY, "approved_create_event_required")
    need(not any(os.environ.get(name) for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE", "PYTHONHTTPSVERIFY",
                                                   "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")), "custom_tls_environment_refused")
    started = os.environ.get("QUANTOPSY_ARTIFACT_JOB_STARTED_MS", "")
    need(len(started) == 13 and started.isascii() and started.isdigit(), "job_clock_required")
    started, now = int(started) / 1000, time.time()
    need(started <= now and now - started < 1200, "job_clock_invalid")
    allowed = min(900, started + 1200 - 90 - 30 - now)
    need(allowed >= 300, "job_entry_window_unavailable")
    _c.work_end = time.monotonic() + allowed
    _c.report["timing"] = {"earliestStepAtUtc": datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
                            "setupElapsedMs": round((now - started) * 1000), "admittedWorkMs": round(allowed * 1000),
                            "cleanupReservedMs": 90000, "marginReservedMs": 30000}
    home = Path(os.environ.get("HOME", ""))
    need(home.is_absolute() and home.is_dir(), "original_home_required")
    parent = Path(os.environ.get("RUNNER_TEMP", "")).resolve(strict=True)
    need(parent.is_dir() and parent.is_absolute() and _c.repo != parent and _c.repo not in parent.parents, "runner_temp_required")
    os.umask(0o077)
    _c.root = Path(tempfile.mkdtemp(prefix="qb-", dir=parent))
    _c.root.chmod(0o700)
    value = _c.root.lstat()
    _c.identity = value.st_dev, value.st_ino
    _c.nonce = secrets.token_hex(16)
    fd = os.open(_c.root / ".owner", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(_c.nonce.encode())
    value = (_c.root / ".owner").lstat()
    _c.marker_identity = value.st_dev, value.st_ino
    free = shutil.disk_usage(_c.root).free
    _c.report["storageAdmission"] = {"freeBytes": free, "minimumBytes": 6 * 1024**3, "reserveBytes": 2 * 1024**3}
    need(free >= 6 * 1024**3, "runner_initial_space_insufficient")
    for name in ("models", "tmp", "output", "config", "cache", "data", "state", "helpers", "downloads", "upstream", "build"):
        (_c.root / name).mkdir(mode=0o700)
        admit_directory(name)
    python = Path(sys.executable).resolve(strict=True)
    _c.env = {"PATH": os.pathsep.join([str(python.parent), "/usr/local/bin", "/usr/bin", "/bin"]),
              "HOME": str(home), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC", "CI": "true",
              "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "TMPDIR": str(_c.root / "tmp"),
              "TMP": str(_c.root / "tmp"), "TEMP": str(_c.root / "tmp"), "XDG_CONFIG_HOME": str(_c.root / "config"),
              "XDG_CACHE_HOME": str(_c.root / "cache"), "XDG_DATA_HOME": str(_c.root / "data"), "XDG_STATE_HOME": str(_c.root / "state"),
              "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
              "SOURCE_COMMIT": _c.source_commit, "QUANTOPSY_ARTIFACT_RUN_ROOT": str(_c.root), "QUANTOPSY_ARTIFACT_NONCE": _c.nonce,
              "QUANTOPSY_ARTIFACT_BINDINGS_SHA256": _c.HELPER_SHA256}
    for folder in Path("/proc").iterdir():
        if folder.name.isdigit():
            try:
                row = _c.proc_info(int(folder.name))
                _c.baseline[row["pid"]] = row["startTime"]
            except (FileNotFoundError, ProcessLookupError):
                pass
    _c.report["python"] = {"version": sys.version, "bytes": python.stat().st_size,
                            "sha256": _c.sha(_c.read(python, 256 * 1024**2)), "installedByPacket": False}
    storage_sample("initial")
    source_guard("before")
    for source, target, expected in ((_c.INVENTORY_HELPER, "helpers/public_gguf_inventory.py", _c.INVENTORY_SHA256),
                                     (_c.HELPER, "helpers/public_gguf_v2_bindings.py", _c.HELPER_SHA256)):
        data = _c.read(_c.repo / source, 128 * 1024)
        need(_c.sha(data) == expected, "independent_helper_source_changed")
        _c.owned_write(target, data)
    _c.copied_helper = _c.root / "helpers/public_gguf_v2_bindings.py"
    _c.report["homePreserved"] = _c.env["HOME"] == str(home)


def fixture_checks():
    """Carry the exact native v3 qualification; execute no fixture case."""
    raw = _c.read(_c.repo / PRIOR_RECEIPT)
    need(_c.sha(raw) == PRIOR_RECEIPT_SHA256, "prior_qualification_digest_changed")
    qualification = _c.strict_json(raw)
    need(qualification["status"] == "qualified-for-authored-fixture-scope"
         and qualification["acceptedCaseCount"] == 74
         and qualification["single74PassInvocation"] is False
         and qualification["originalFailedVerdictsRetained"] is True, "prior_qualification_shape")
    directory = _c.repo / "docs/results/2026/11-13"
    first = _c.strict_json(_c.read(directory / "profile-v3-checks-attempt-1.json"))
    second = _c.strict_json(_c.read(directory / "profile-v3-checks-attempt-2.json"))
    need(first["status"] == "observer_failed" and first["command"]["exitObserved"]["code"] == 1
         and first["command"]["closed"]["code"] == 1 and second["command"]["exitObserved"]["code"] == 0
         and second["command"]["closed"]["code"] == 0, "prior_actual_child_outcomes_changed")
    expected = []
    for module, class_name in (("tests/integration/test_profile_v3.py", "ProfileV3IntegrationTier2Tests"),
                               ("tests/integration/test_profile_v2.py", "ProfileV2IntegrationTier2Tests"),
                               ("tests/test_cli.py", "CliTests"), ("tests/test_behavior.py", "BehaviorComparisonTier2Tests")):
        tree = ast.parse(_c.read(_c.repo / module, 128 * 1024))
        prefix = module.removesuffix(".py").replace("/", ".") + "." + class_name + "."
        expected.extend(prefix + node.name for item in tree.body
                        if isinstance(item, ast.ClassDef) and item.name == class_name
                        for node in item.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                        and (module != "tests/test_behavior.py" or "manifest" in node.name))
    observed = first["cases"]["observed"]
    failed = [row["id"] for row in observed if row["status"] == "FAIL"]
    passed = [row for row in observed if row["status"] == "ok"]
    completed = second["cases"]["observed"]
    need(len(expected) == len(set(expected)) == len(observed) == 74
         and sorted(row["id"] for row in observed) == sorted(expected)
         and len(passed) == 72 and len(failed) == len(completed) == 2
         and sorted(row["id"] for row in completed) == sorted(failed)
         and all(row["status"] == "ok" for row in completed)
         and qualification["acceptedCases"] == passed + completed, "prior_case_coverage_changed")
    earlier = {row["path"]: row for row in first["source"]["before"]}
    accepted = {row["path"]: row for row in second["source"]["before"]}
    current = {row["path"]: row for row in _c.source_vector}
    test_path = "tests/integration/test_profile_v3.py"
    need(set(earlier) == set(accepted) and len(accepted) == 54
         and all(earlier[path] == row for path, row in accepted.items() if path != test_path),
         "prior_72_case_carry_changed")
    old_test = _c.read(directory / "profile-v3-test-source-attempt-1.py.txt", 128 * 1024)
    before = b'self.refused("repair_invalid_values", self.plan, reference, quantized)'
    after = b'self.refused("gguf_nonfinite_weight", self.plan, reference, quantized)'
    need(_c.sha(old_test) == earlier[test_path]["sha256"] and old_test.count(before) == 2
         and old_test.replace(before, after) == _c.read(_c.repo / test_path, 128 * 1024),
         "prior_two_literal_only_repair_changed")
    fixed = sorted(path for path in accepted if path.startswith(("quantopsy/", "tests/")) and path != "tests/README.md")
    need(fixed and all(path in current and current[path] == accepted[path] for path in fixed),
         "qualified_product_or_test_input_changed")
    need({path for path in current if path.startswith(("quantopsy/", "tests/"))
          and path not in (CONTROLLER, DATA_HELPER, "tests/README.md")} == set(fixed),
         "qualified_runtime_input_set_changed")
    for report in (first, second):
        need(report["source"]["unchanged"] is True and report["source"]["indexUnchanged"] is True
             and report["command"]["stop"] is None and report["command"]["launchError"] is None,
             "prior_source_guard_or_command_failed")
    _c.report["fixture"] = {"accepted": True, "executed": False, "expected": 74,
        "identities": sorted(expected), "passedFromFirst": 72, "completedFromSecond": 2,
        "single74PassInvocation": False, "matchedSourceInputs": fixed,
        "evidence": {"file": PRIOR_RECEIPT, "sha256": PRIOR_RECEIPT_SHA256,
                     "qualifiedSourceCommit": "27ea65a5d0d7e65cae18f6a8380e6e8af78f09e3",
                     "runtime": "native Python3.14.7", "thisInvocationIsNotARepeat": True}}


def download_source():
    """Observe, rather than invent, the archive hash from the official TLS URL."""
    global _archive_owner
    _c.owner()
    row = {"url": ARCHIVE_URL, "requestedRevision": LLAMA_REVISION, "complete": False,
           "predeclaredArchiveSha256": None, "archiveIdentityAttestation": "official_url_and_verified_tls_only",
           "receivedBytes": 0, "bodyEOFObserved": False, "responseClosed": False, "fileDescriptorClosed": False}
    _c.report["engineSource"] = row
    target = _c.root / "downloads/llama-source.tar.gz"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    created = _c.file_identity(os.fstat(fd))
    _archive_owner = {"relativePath": "downloads/llama-source.tar.gz", "createdDevice": created["device"],
                      "createdInode": created["inode"], "bytes": 0, "sha256": _c.sha(b"")}
    response = None
    digest = hashlib.sha256()
    started = time.monotonic()
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError("archive_download_deadline")))
    signal.setitimer(signal.ITIMER_REAL, min(120, max(0.1, _c.work_end - started)))
    try:
        addresses = socket.getaddrinfo("codeload.github.com", 443, type=socket.SOCK_STREAM)
        need(addresses and all(_c.ipaddress.ip_address(item[4][0]).is_global for item in addresses), "archive_dns_not_public")
        context = ssl.create_default_context()
        need(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED and context.keylog_filename is None, "verified_tls_required")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _c.NoRedirect(), urllib.request.HTTPSHandler(context=context))
        request = urllib.request.Request(ARCHIVE_URL, headers={"Accept-Encoding": "identity", "User-Agent": "Quantopsy-behavioral-inference/1"})
        response = opener.open(request, timeout=15)
        need(response.code == 200 and response.geturl() == ARCHIVE_URL
             and response.headers.get("Content-Encoding", "identity").strip().lower() in ("", "identity"), "archive_http_refusal")
        length = response.headers.get("Content-Length")
        if length is not None:
            need(length.isascii() and length.isdigit() and 0 < int(length) <= MAX_ARCHIVE, "archive_declared_byte_bound")
            row["declaredBytes"] = int(length)
        while True:
            need(not _c.interrupted and time.monotonic() < _c.work_end and time.monotonic() - started < 120, "archive_download_deadline")
            chunk = response.read(min(1024**2, MAX_ARCHIVE - row["receivedBytes"]) or 1)
            if not chunk:
                row["bodyEOFObserved"] = True
                break
            need(row["receivedBytes"] + len(chunk) <= MAX_ARCHIVE, "archive_byte_bound")
            offset = 0
            while offset < len(chunk):
                written = os.write(fd, chunk[offset:])
                need(written > 0, "archive_write_incomplete")
                digest.update(chunk[offset:offset + written])
                offset += written
                row["receivedBytes"] += written
        need(row["receivedBytes"] > 0 and (length is None or row["receivedBytes"] == int(length)), "archive_length_mismatch")
        os.fsync(fd)
        need(_c.file_identity(os.fstat(fd)) == _c.file_identity(target.lstat())
             and _c.file_identity(target.lstat())["inode"] == created["inode"], "archive_file_replaced")
        row.update({"sha256": digest.hexdigest(), "complete": True})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if response is not None:
            response.close()
        row["responseClosed"] = True
        os.close(fd)
        row["fileDescriptorClosed"] = True
        row["observedPartialSha256"] = digest.hexdigest()
        _archive_owner.update({"bytes": row["receivedBytes"], "sha256": digest.hexdigest()})
        current = None
        if time.monotonic() < _c.work_end:
            current = admit_file("downloads/llama-source.tar.gz")
            need(current["device"] == created["device"] and current["inode"] == created["inode"], "archive_file_replaced")
    need(row["complete"] and current["sha256"] == row["sha256"], "archive_download_incomplete")
    storage_sample("archive_downloaded")


def extract_source():
    archive = fingerprint(_c.root / "downloads/llama-source.tar.gz")
    total = count = 0
    blobs = {}
    with tarfile.open(_c.root / archive["relativePath"], mode="r:gz") as source:
        for member in source:
            need(not _c.interrupted and time.monotonic() < _c.work_end, "archive_extract_deadline")
            path = PurePosixPath(member.name)
            need(not path.is_absolute() and path.parts and path.parts[0] == ARCHIVE_ROOT
                 and all(part not in ("", ".", "..") for part in path.parts) and "\\" not in member.name
                 and len(member.name.encode()) <= 1024 and (member.isfile() or member.isdir()), "archive_entry_refused")
            count += 1
            total += member.size if member.isfile() else 0
            need(count <= MAX_FILES and total <= MAX_EXTRACTED, "archive_expansion_bound")
            target = _c.root / "upstream" / Path(*path.parts)
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            need(member.size >= 0 and not os.path.lexists(target), "archive_duplicate_or_size")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            body = source.extractfile(member)
            need(body is not None, "archive_member_unreadable")
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700 if member.mode & 0o111 else 0o600)
            git_digest = hashlib.sha1(b"blob " + str(member.size).encode() + b"\0")
            try:
                with os.fdopen(fd, "wb") as output:
                    remaining = member.size
                    while remaining:
                        need(time.monotonic() < _c.work_end, "archive_extract_deadline")
                        chunk = body.read(min(1024**2, remaining))
                        need(bool(chunk), "archive_member_truncated")
                        output.write(chunk)
                        git_digest.update(chunk)
                        remaining -= len(chunk)
                    need(body.read(1) == b"", "archive_member_length")
            finally:
                body.close()
            relative = str(PurePosixPath(*path.parts[1:]))
            need(relative not in blobs, "archive_duplicate_git_path")
            actual_blob = native_blob(target)
            need(actual_blob == git_digest.hexdigest(), "extracted_file_differs_from_archive_git_blob")
            blobs[relative] = ("100755" if member.mode & 0o111 else "100644", actual_blob)
    need(fingerprint(_c.root / archive["relativePath"]) == archive, "archive_changed_during_extract")
    need(len(blobs) == 1240 and total == 68_926_775, "upstream_complete_blob_inventory")
    tree = reconstruct_git_tree(blobs)
    need(tree == LLAMA_TREE, "upstream_git_tree_identity_mismatch")
    _c.report["engineSource"].update({"extractedEntries": count, "extractedBytes": total, "tree": admit_tree("upstream"),
        "gitObjectIdentity": "full_git_tree_reconstructed_and_matched_reviewed_commit_tree", "gitTreeSha1": tree,
        "trustedCommitTreeSha1": LLAMA_TREE, "blobCount": len(blobs), "pinnedUnfetchedGitlinks": list(_data.UPSTREAM_GITLINKS),
        "archiveIdentityAttestation": "official_tls_archive_and_full_reviewed_git_tree"})
    need((_c.root / "upstream" / ARCHIVE_ROOT / "include/llama.h").is_file(), "upstream_api_header_missing")


def native_blob(path):
    before = fingerprint(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest = hashlib.sha1(b"blob " + str(before["bytes"]).encode("ascii") + b"\0")
    try:
        need(_c.file_identity(os.fstat(fd)) == {key: before[key] for key in
             ("device", "inode", "uid", "mode", "bytes", "mtimeNs", "ctimeNs")}, "git_blob_changed_at_open")
        remaining = before["bytes"]
        while remaining:
            need(time.monotonic() < _c.work_end and not _c.interrupted, "git_blob_hash_deadline")
            chunk = os.read(fd, min(1024**2, remaining))
            need(bool(chunk), "git_blob_truncated")
            remaining -= len(chunk)
            digest.update(chunk)
        need(os.read(fd, 1) == b"" and fingerprint(path) == before, "git_blob_file_changed")
    finally:
        os.close(fd)
    return digest.hexdigest()


def reconstruct_git_tree(blobs):
    """Rebuild native Git objects, including one explicitly pinned gitlink."""
    root = {}
    for path, entry in blobs.items():
        parts = path.split("/")
        current = root
        for part in parts[:-1]:
            current = current.setdefault(part, {})
            need(type(current) is dict, "git_tree_path_collision")
        need(parts[-1] not in current, "git_tree_path_collision")
        current[parts[-1]] = entry
    expected_links = [{"path": "ggml/src/ggml-kompute/kompute", "mode": "160000",
                       "sha1": "4565194ed7c32d1d2efa32ceab4d3c6cae006306"}]
    need(list(_data.UPSTREAM_GITLINKS) == expected_links, "pinned_gitlink_contract_changed")
    for link in expected_links:
        path = _c.root / "upstream" / ARCHIVE_ROOT / link["path"]
        need(not path.exists() or (path.is_dir() and not any(path.iterdir())), "unfetched_gitlink_not_empty")
        parts = link["path"].split("/")
        current = root
        for part in parts[:-1]:
            need(part in current and type(current[part]) is dict, "gitlink_parent_missing")
            current = current[part]
        need(parts[-1] not in current, "gitlink_archive_content_refused")
        current[parts[-1]] = (link["mode"], link["sha1"])

    def digest_tree(current):
        encoded = bytearray()
        for name in sorted(current, key=lambda item: (item + ("/" if type(current[item]) is dict else "")).encode("utf-8")):
            child = current[name]
            mode, digest = ("40000", digest_tree(child)) if type(child) is dict else child
            encoded.extend(mode.encode("ascii") + b" " + name.encode("utf-8") + b"\0" + bytes.fromhex(digest))
        return hashlib.sha1(b"tree " + str(len(encoded)).encode("ascii") + b"\0" + encoded).hexdigest()
    return digest_tree(root)


def build_engine():
    global _binary
    source_guard("before_build")
    admit_tree("upstream")
    for tool in ("cmake", "c++"):
        path = shutil.which(tool, path=_c.env["PATH"])
        need(path is not None, "preinstalled_build_tool_missing")
        actual = Path(path).resolve(strict=True)
        _c.report.setdefault("buildTools", []).append({"name": tool, "resolvedBasename": actual.name,
                                                       "bytes": actual.stat().st_size, "sha256": _c.sha(_c.read(actual, 256 * 1024**2))})
        _c.run("preinstalled_" + tool.replace("+", "p") + "_version", [path, "--version"], 10)
    configure = ["cmake", "-S", str(_c.repo / "tests/helpers/behavioral_engine"), "-B", str(_c.root / "build"),
                 "-DQUANTOPSY_LLAMA_SOURCE=" + str(_c.root / "upstream" / ARCHIVE_ROOT),
                 "-DQUANTOPSY_SCORER_SOURCE=" + str(_c.repo / SCORER), "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                 "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON"]
    try:
        _c.run("configure_pinned_cpu_scorer", configure, 90, cwd=_c.root / "build")
        _c.run("build_pinned_cpu_scorer", ["cmake", "--build", str(_c.root / "build"), "--target", "quantopsy-behavioral-scorer", "--parallel", "2"],
               360, cwd=_c.root / "build")
    finally:
        if all(process.poll() is not None and _c.group_gone(process.pid) for process in _c.processes):
            _c.report["engineBuildTree"] = admit_tree("build", mutable=True)
    admit_tree("upstream")
    _binary = fingerprint(_c.root / "build/quantopsy-behavioral-scorer")
    require_entry(_binary["mode"] in ("0700", "0755"), "scorer_executable_mode",
                  _c.root / _binary["relativePath"], (_c.root / _binary["relativePath"]).lstat())
    _c.report["engineBinary"] = _binary
    source_guard("after_build")


OBSERVE_CANDIDATE = r'''
import hashlib,json,os,sys,types
from pathlib import Path
root=Path(os.environ['QUANTOPSY_ARTIFACT_RUN_ROOT'])
path=root/'helpers/public_gguf_v2_bindings.py'
raw=path.read_bytes()
assert hashlib.sha256(raw).hexdigest()==os.environ['QUANTOPSY_ARTIFACT_BINDINGS_SHA256']
m=types.ModuleType('quantopsy_fixed_binding_observer');m.__file__=str(path)
exec(compile(raw,str(path),'exec'),m.__dict__)
opened=[];rows=[]
original,marker=root.lstat(),(root/'.owner').lstat()
m.root_guard(root,os.environ['QUANTOPSY_ARTIFACT_NONCE'],original,marker)
try:
    observer_file=m.HeldFile(root,'helpers/public_gguf_inventory.py',131072,rows,opened,size=m.INVENTORY_BYTES,sha256=m.INVENTORY_SHA256)
    observer=m.load_observer(observer_file)
    relative=sys.argv[1]
    assert relative in ('output/candidate-v2/candidate.gguf','output/candidate-v3/candidate.gguf')
    candidate=m.HeldFile(root,relative,100663296,rows,opened)
    inventory=m.observe_payloads(observer,candidate)
finally:
    for held in opened:held.verify()
    for held in reversed(opened):held.close()
m.root_guard(root,os.environ['QUANTOPSY_ARTIFACT_NONCE'],original,marker)
assert all(row['unchanged'] and row['closed'] for row in rows)
print('QUANTOPSY_BEHAVIOR_CANDIDATE_INVENTORY='+json.dumps({'inventory':inventory,'files':rows},sort_keys=True,separators=(',',':')))
'''


def selection_rows(pair, profile):
    protocol = _data.protocol_contract(_c.repo)
    reference = {row["name"]: row for row in pair["inputs"][0]["inventory"]["tensors"]}
    quantized = {row["name"]: row for row in pair["inputs"][1]["inventory"]["tensors"]}
    names = list(_data.LEGACY_TENSORS if profile == "v2" else _data.SELECTED_TENSORS)
    need(profile in ("v2", "v3"), "candidate_profile_required")
    target = protocol["candidate_strategy"]["legacy_internal_matrices" if profile == "v2" else "embedding_extension"]
    rows = []
    for name in names:
        a, b = reference[name], quantized[name]
        need(a["ggmlType"] == 0 and b["ggmlType"] == 2 and a["dimensions"] == b["dimensions"]
             and len(a["dimensions"]) == 2, "selection_type_or_shape")
        rows.append({"name": name, "dimensions": a["dimensions"], "elements": a["elements"], "reference_kind": 0,
                     "quantized_kind": 2, "reference_sha256": a["payloadSha256"], "quantized_sha256": b["payloadSha256"],
                     "reference_bytes": a["payloadBytes"], "quantized_bytes": b["payloadBytes"]})
    need(len(rows) == target["count"] and sum(row["elements"] for row in rows) == target["selected_elements"]
         and sum(row["reference_bytes"] - row["quantized_bytes"] for row in rows) == target["added_weight_bytes"],
         "predeclared_selection_work_or_growth")
    embedding = protocol["candidate_strategy"]["embedding_extension"]
    a, b = reference["token_embd.weight"], quantized["token_embd.weight"]
    output = quantized["output.weight"]
    head = protocol["candidate_strategy"]["preserved_output_head"]
    need(a["dimensions"] == b["dimensions"] == embedding["dimensions"]
         and a["payloadSha256"] == embedding["reference_payload_sha256"]
         and b["payloadSha256"] == embedding["quantized_payload_sha256"]
         and output["ggmlType"] == 8 and output["payloadBytes"] == head["bytes"]
         and output["payloadSha256"] == head["sha256"], "embedding_or_output_binding_changed")
    _c.report.setdefault("manualSelections", {})[profile] = {
        "names": names, "elements": target["selected_elements"], "addedWeightBytes": target["added_weight_bytes"],
        "frozenBeforeCandidateOutputs": True, "automaticSelection": False}
    return rows, target


def candidate_bundle(pair, selected, target, profile):
    directory = "output/candidate-" + profile
    plan_path = "output/repair-plan-" + profile + ".json"
    args = ["plan", _c.root / "models" / _c.FIXTURES[0]["name"], _c.root / "models" / _c.FIXTURES[1]["name"], "--profile", profile]
    for row in selected:
        args.extend(["--tensor", row["name"]])
    args.extend(["--provenance", _c.root / "output/provenance.json", "--max-added-bytes", MAX_ADDED_BYTES,
                 "--max-elements", target["selected_elements"], "--output", _c.root / plan_path])
    _c.product_command("plan_" + profile + "_" + str(len(selected)) + "_explicit_tensors", args, seconds=180, outputs=True)
    plan, plan_raw = _c.read_owned_json(plan_path)
    expected = {"schema_version": int(profile[1:]), "kind": "quantopsy.repair-plan", "architecture": "llama", "provenance": _c.PROVENANCE,
                "reference": {"name": _c.FIXTURES[0]["name"], "sha256": _c.FIXTURES[0]["sha256"], "size": _c.FIXTURES[0]["bytes"]},
                "quantized": {"name": _c.FIXTURES[1]["name"], "sha256": _c.FIXTURES[1]["sha256"], "size": _c.FIXTURES[1]["bytes"]},
                "selected": selected, "added_weight_bytes": target["added_weight_bytes"], "max_added_bytes": MAX_ADDED_BYTES,
                "max_elements": target["selected_elements"], "lineage": "operator_declared_compatibility_checked",
                "behavioral_validation": "not_run"}
    need(plan == expected and plan_raw == _c.canonical(expected), "plan_does_not_match_predeclared_strategy")
    cli, cli_raw = _c.product_command("repair_" + profile + "_" + str(len(selected)) + "_explicit_tensors",
        ["repair", _c.root / "models" / _c.FIXTURES[0]["name"], _c.root / "models" / _c.FIXTURES[1]["name"],
         "--profile", profile, "--plan", _c.root / plan_path, "--output-dir", _c.root / directory,
         "--reserve-bytes", RESERVE_BYTES], seconds=180, outputs=True)
    manifest, manifest_raw = _c.read_owned_json(directory + "/repair.json")
    bundled, bundled_raw = _c.read_owned_json(directory + "/plan.json")
    need(cli == manifest and cli_raw == manifest_raw and bundled == plan and bundled_raw == plan_raw,
         "manifest_or_plan_originals_differ")
    current = fingerprint(_c.root / directory / "candidate.gguf")
    need(current["bytes"] == target["candidate_bytes"] and current["bytes"] <= 96 * 1024**2, "candidate_file_bound_or_size")
    if profile == "v2":
        need(current["sha256"] == target["candidate_sha256"], "legacy_candidate_not_byte_identical_to_original")
    out, err, command = _c.run("independent_" + profile + "_57_payload_inventory",
        [sys.executable, "-I", "-B", "-S", "-c", OBSERVE_CANDIDATE, directory + "/candidate.gguf"], 100)
    prefix = b"QUANTOPSY_BEHAVIOR_CANDIDATE_INVENTORY="
    need(err == b"" and out.startswith(prefix) and out.endswith(b"\n") and out.count(prefix) == 1,
         "candidate_inventory_original_marker")
    value = _c.strict_json(out[len(prefix):-1])
    inventory = value["inventory"]
    _c.validate_inventory(_c.root / current["relativePath"], current["bytes"], inventory)
    for row in value["files"]:
        original = fingerprint(_c.root / row["relativePath"])
        fields = {key: original[key] for key in ("device", "inode", "uid", "mode", "bytes", "mtimeNs", "ctimeNs")}
        need(row["before"] == row["after"] == fields
             and row["sha256Before"] == row["sha256After"] == original["sha256"]
             and row["closed"] is True and row["unchanged"] is True, "candidate_observer_file_identity")
    reference = {row["name"]: row for row in pair["inputs"][0]["inventory"]["tensors"]}
    q_inventory = pair["inputs"][1]["inventory"]
    quantized = {row["name"]: row for row in q_inventory["tensors"]}
    need([row["name"] for row in inventory["tensors"]] == list(quantized), "candidate_tensor_order")
    expected_metadata = [row for row in q_inventory["metadata"] if row["key"] != "general.file_type"]
    need([row["key"] for row in inventory["metadata"]] == [row["key"] for row in expected_metadata]
         and all(_c.metadata_signature(a) == _c.metadata_signature(b)
                 for a, b in zip(inventory["metadata"], expected_metadata)), "candidate_metadata_preservation")
    payloads = []
    selected_names = {row["name"] for row in selected}
    for actual in inventory["tensors"]:
        name = actual["name"]
        source = reference[name] if name in selected_names else quantized[name]
        need((actual["dimensions"], actual["ggmlType"], actual["payloadBytes"], actual["payloadSha256"])
             == (source["dimensions"], source["ggmlType"], source["payloadBytes"], source["payloadSha256"]),
             "candidate_prescribed_payload_mismatch")
        payloads.append({"name": name, "source": "reference" if name in selected_names else "quantized",
                         "bytes": actual["payloadBytes"], "sha256": actual["payloadSha256"]})
    expected_manifest = {key: value for key, value in plan.items() if key not in ("kind", "schema_version")}
    expected_manifest.update({"schema_version": int(profile[1:]), "kind": "quantopsy.repair-result",
        "plan_sha256": _c.sha(plan_raw),
        "output": {"name": "candidate.gguf", "sha256": current["sha256"], "size": current["bytes"]},
        "reserve_bytes": RESERVE_BYTES,
        "metadata_omissions": ["general.file_type"] if len(expected_metadata) != len(q_inventory["metadata"]) else [],
        "verification": {"container_reparsed": True, "metadata_preserved": True,
                         "tensor_payloads_verified": True, "source_hashes_verified": True}})
    need(manifest == expected_manifest and manifest_raw == _c.canonical(expected_manifest)
         and len(payloads) == 57 and sum(row["source"] == "reference" for row in payloads) == len(selected),
         "candidate_manifest_or_binding_count")
    _c.report.setdefault("candidates", {})[profile] = {
        "plan": plan, "planSha256": _c.sha(plan_raw), "manifest": manifest, "manifestSha256": _c.sha(manifest_raw),
        "independentPayloads": payloads, "inventoryOriginalCommand": command["label"], "candidateIdentity": current}
    return inventory


def tokenizer_hash(inventory):
    fields = ("key", "kind", "arrayKind", "count", "encodedBytes", "encodedSha256")
    rows = [{key: row[key] for key in fields} for row in inventory["metadata"] if row["key"].startswith("tokenizer.")]
    need(rows, "tokenizer_identity_missing")
    return _c.sha(_data.canonical(sorted(rows, key=lambda row: row["key"])))


def freeze_measurements(pair, candidate_inventories):
    tokenizers = [tokenizer_hash(row["inventory"]) for row in pair["inputs"]]
    tokenizers.extend(tokenizer_hash(candidate_inventories[profile]) for profile in ("v2", "v3"))
    need(len(tokenizers) == 4 and len(set(tokenizers)) == 1, "tokenizer_four_role_mismatch")
    manifest = _data.engine_manifest(_c.repo)
    manifest_raw = _data.canonical(manifest)
    config = _data.engine_config(manifest_raw, tokenizers[0])
    config_raw = _data.canonical(config)
    suite = _data.frozen_suite(config_raw)
    protocol = _data.protocol_contract(_c.repo)
    need(suite["metrics"] == METRICS == protocol["metrics"] and suite["tasks"] == protocol["tasks"]
         and len(suite["tasks"]) == 8, "frozen_suite_protocol_changed")
    suite_raw = _data.canonical(suite)
    models = {}
    for profile in ("v2", "v3"):
        models[profile] = {"schema_version": 1, "kind": "quantopsy.behavior-models"}
        for role, relative in zip(("reference", "quantized", "candidate"),
            ("models/" + _c.FIXTURES[0]["name"], "models/" + _c.FIXTURES[1]["name"],
             "output/candidate-" + profile + "/candidate.gguf")):
            row = fingerprint(_c.root / relative)
            models[profile][role] = {"name": Path(relative).name, "size": row["bytes"], "sha256": row["sha256"]}
            _frozen[relative] = row
    need(models["v2"]["reference"] == models["v3"]["reference"]
         and models["v2"]["quantized"] == models["v3"]["quantized"], "comparison_triplets_do_not_share_baselines")
    records = {"output/engine-manifest.json": manifest_raw, "output/config.json": config_raw,
               "output/suite.json": suite_raw, "output/tasks.tsv": _data.encode_tasks(suite),
               "output/models-v2.json": _c.canonical(models["v2"]), "output/models-v3.json": _c.canonical(models["v3"]),
               "output/predeclared-protocol.json": _c.read(_c.repo / PROTOCOL)}
    for relative, raw in records.items():
        _c.owned_write(relative, raw)
        _frozen[relative] = fingerprint(_c.root / relative)
    _frozen["build/quantopsy-behavioral-scorer"] = _binary
    originals = {relative: _c.capture(raw, len(raw), _c.sha(raw), True) for relative, raw in records.items()}
    need(all(row["complete"] and row["retained"] for row in originals.values()), "frozen_original_retention_failed")
    _c.report["measurementFreeze"] = {"beforeAnyInference": True, "utc": _c.utc(), "files": list(_frozen.values()),
        "originals": originals, "metrics": METRICS, "taskCount": 8, "roleCount": 4, "tokenizerSha256": tokenizers[0],
        "engineManifest": manifest, "configuration": config, "protocolSha256": PROTOCOL_SHA256,
        "sourceVectorSha256": _c.sha(_c.canonical(_c.source_vector))}
    return suite_raw, config_raw, models


def frozen_guard(label):
    _c.verify_inputs(label)
    for relative, previous in _frozen.items():
        need(fingerprint(_c.root / relative) == previous, "frozen_measurement_input_changed")
    _c.verify_generated(label)


def inference(suite_raw, config_raw, models):
    details = {}
    roles = (("reference", "reference", "models/" + _c.FIXTURES[0]["name"], "v2"),
             ("quantized", "quantized", "models/" + _c.FIXTURES[1]["name"], "v2"),
             ("candidate-v2", "candidate", "output/candidate-v2/candidate.gguf", "v2"),
             ("candidate-v3", "candidate", "output/candidate-v3/candidate.gguf", "v3"))
    for measurement_role, record_role, relative, profile in roles:
        frozen_guard("before_inference_" + measurement_role)
        out, err, command = _c.run("real_cpu_forward_" + measurement_role,
            [_c.root / "build/quantopsy-behavioral-scorer", _c.root / relative, _c.root / "output/tasks.tsv"],
            90, cwd=_c.root / "output")
        collected = _data.collect_run(out, record_role, suite_raw, config_raw, models[profile][record_role])
        need(set(collected) == {"run", "details"}, "scorer_collection_shape")
        details[measurement_role] = collected["details"]
        run_raw = _c.canonical(collected["run"])
        _c.owned_write("output/" + measurement_role + "-run.json", run_raw)
        _c.owned_write("output/" + measurement_role + "-token-details.json", _c.canonical(collected["details"]))
        _c.report.setdefault("inferenceRuns", []).append({"measurementRole": measurement_role,
            "recordRole": record_role, "originalCommand": command["label"], "run": collected["run"],
            "runOriginal": _c.capture(run_raw, len(run_raw), _c.sha(run_raw), True),
            "detailsSha256": _c.sha(_c.canonical(collected["details"])),
            "stderrOriginalRetained": command["stderr"]["complete"], "durationMs": command["durationMs"]})
        frozen_guard("after_inference_" + measurement_role)
    comparisons = {}
    for profile in ("v2", "v3"):
        _data.compare_tokenization({"reference": details["reference"], "quantized": details["quantized"],
                                    "candidate": details["candidate-" + profile]})
        args = ["compare-behavior", "--suite", _c.root / "output/suite.json", "--config", _c.root / "output/config.json",
                "--models", _c.root / ("output/models-" + profile + ".json"),
                "--repair-manifest", _c.root / ("output/candidate-" + profile + "/repair.json"),
                "--reference-run", _c.root / "output/reference-run.json", "--quantized-run", _c.root / "output/quantized-run.json",
                "--candidate-run", _c.root / ("output/candidate-" + profile + "-run.json"),
                "--output", _c.root / ("output/comparison-" + profile + ".json")]
        _c.product_command("compare_frozen_records_" + profile, args, outputs=True)
        comparison, raw = _c.read_owned_json("output/comparison-" + profile + ".json")
        need(comparison["acceptance"]["complete_runs"] is True
             and type(comparison["acceptance"]["supplied_recovery_records_pass"]) is bool,
             "comparison_did_not_cover_complete_runs")
        comparisons[profile] = {"value": comparison, "bytes": len(raw), "sha256": _c.sha(raw)}
        frozen_guard("after_comparison_" + profile)
    _c.report["comparisons"] = comparisons
    _c.report["tokenizationAcrossFourRoles"] = "exact_prompt_and_target_token_ids_verified"
    legacy = comparisons["v2"]["value"]["acceptance"]["supplied_recovery_records_pass"]
    embedding = comparisons["v3"]["value"]["acceptance"]["supplied_recovery_records_pass"]
    _c.report["suppliedRecoveryRecordsPass"] = embedding
    _c.report["studyDecision"] = {"legacy42Pass": legacy, "embedding43Pass": embedding,
        "bothPassDoesNotEstablishEmbeddingNecessity": legacy and embedding,
        "generalBehavioralRecoveryAccepted": False, "oldStudyVerdictChanged": False,
        "compression": {profile: {"candidateBytes": models[profile]["candidate"]["size"],
                                 "referenceBytes": models[profile]["reference"]["size"],
                                 "quantizedBytes": models[profile]["quantized"]["size"]}
                        for profile in ("v2", "v3")}}
    admit_tree("upstream")
    admit_tree("build")
    source_guard("after_inference")


def cleanup():
    _c.cleanup_end = min(time.monotonic() + 90, _c.work_end + 90 if _c.work_end is not None else time.monotonic() + 90)
    result = {"passed": False, "signals": [], "failures": []}
    _c.report["cleanup"] = result
    if _c.root is None or _c.identity is None:
        result["rootCreated"] = False
        return
    try:
        source_guard("before_cleanup", cleanup=True)
    except Exception as error:
        result["failures"].append("source_" + _c.finite(error))
    try:
        _c.owner()
        for process in _c.processes:
            if process.poll() is None and process.pid in _c.groups and _c.signal_original_group(process.pid, signal.SIGTERM):
                result["signals"].append({"pid": process.pid, "signal": "SIGTERM"})
        end = min(time.monotonic() + 3, _c.cleanup_end - 10)
        while time.monotonic() < end and any(process.poll() is None for process in _c.processes):
            time.sleep(0.05)
        for process in _c.processes:
            if process.poll() is None and process.pid in _c.groups and _c.signal_original_group(process.pid, signal.SIGKILL):
                result["signals"].append({"pid": process.pid, "signal": "SIGKILL"})
            if process.poll() is None:
                process.wait(timeout=max(0.01, min(1, _c.cleanup_end - time.monotonic())))
        rows, uncertain = _c.owned()
        need(not rows and not uncertain and all(process.poll() is not None for process in _c.processes)
             and all(_c.group_gone(pid) for pid in _c.groups), "process_cleanup_unverified")
        collect_outputs()
        # A closed build may leave legitimate compiler outputs. Only its
        # pre-created tree is mutable; extracted sources and frozen inputs are not.
        admit_tree("build", mutable="build" not in _tree_frozen)
        admit_tree("upstream", mutable="upstream" not in _tree_frozen)
        for row in _c.report["downloads"]:
            partial = row.get("partialFile")
            if partial is not None:
                relative = partial["relativePath"]
                current = fingerprint(_c.root / relative, {"bytes": partial["writtenBytes"], "sha256": partial["writtenSha256"]})
                need((current["device"], current["inode"]) == (partial["createdDevice"], partial["createdInode"]), "download_partial_replaced")
                _c.known_files[relative] = {"createdDevice": current["device"], "createdInode": current["inode"], "file": current}
        if _archive_owner is not None:
            current = fingerprint(_c.root / _archive_owner["relativePath"], _archive_owner)
            need((current["device"], current["inode"]) == (_archive_owner["createdDevice"], _archive_owner["createdInode"]), "archive_partial_replaced")
            _c.known_files[_archive_owner["relativePath"]] = {"createdDevice": current["device"], "createdInode": current["inode"], "file": current}
        for relative, previous in _frozen.items():
            need(fingerprint(_c.root / relative) == previous, "frozen_input_changed_before_cleanup")
        _c.verify_inputs("before_removal")
        _c.verify_generated("before_removal")
        expected = set(_c.known_files) | {".owner"} | {row["relativePath"] for row in _c.tracked_files.values()}
        actual, directories = set(), set()
        for parent, folders, names in os.walk(_c.root, followlinks=False):
            for name in folders:
                relative = str((Path(parent) / name).relative_to(_c.root))
                need(relative in _c.known_directories, "unexpected_directory_preserved")
                admit_directory(relative)
                directories.add(relative)
            for name in names:
                relative = str((Path(parent) / name).relative_to(_c.root))
                need(relative in expected, "unexpected_file_preserved")
                actual.add(relative)
        need(actual == expected and directories == set(_c.known_directories), "owned_entry_set_changed")
        _c.owner()
        storage_sample("before_removal")
        result["ownedFilesVerified"] = len(actual)
        result["ownedDirectoriesVerified"] = len(directories)
        result["ownedFileVectorSha256"] = _c.sha(_c.canonical(sorted([row["file"] for row in _c.known_files.values()], key=lambda row: row["relativePath"])))
        shutil.rmtree(_c.root)
        need(not _c.root.exists(), "scratch_still_present")
        _c.report["scratchCleanup"] = "passed"
        source_guard("after_cleanup", cleanup=True, post_scratch=True)
        rows, uncertain = _c.owned()
        need(not rows and not uncertain and all(_c.group_gone(pid) for pid in _c.groups), "post_cleanup_process_remaining")
        result.update({"allOriginalGroupsAbsent": True, "allChildrenReaped": True, "remaining": rows, "uncertain": uncertain,
                       "attributedPidStarttimeObservations": list(_c.attributed.values()), "passed": not result["failures"] and not result["signals"]})
    except Exception as error:
        result["failures"].append(_c.finite(error))
        _c.report.setdefault("scratchCleanup", "unverified")


def install_support():
    global _c, _data
    repo = Path(__file__).resolve().parents[2]
    need(Path.cwd().resolve() == repo, "source_working_directory_required")
    _c = load_fixed(repo / SUPPORT, "quantopsy_closed_v2_operational_support", SUPPORT_SHA256)
    _c.repo, _c.REPOSITORY, _c.BRANCH, _c.ACTIVE, _c.CONTROLLER = repo, REPOSITORY, BRANCH, ACTIVE, CONTROLLER
    _c.STORAGE_LIMIT = MAX_STORAGE
    _c.ZIP_LIMIT, _c.ALL_ZIP_LIMIT, _c.REPORT_LIMIT = 1024**2, 8 * 1024**2, 16 * 1024**2
    _c.storage_sample, _c.fingerprint, _c.collect_outputs, _c.guard = storage_sample, fingerprint, collect_outputs, source_guard
    _c.verify_generated = verify_generated
    _c.preflight, _c.cleanup = preflight, cleanup
    original_popen = _c.subprocess.Popen

    def child_limits():
        for kind, maximum in ((resource.RLIMIT_AS, 2 * 1024**3), (resource.RLIMIT_CPU, 900),
                              (resource.RLIMIT_FSIZE, MAX_BUILD), (resource.RLIMIT_NOFILE, 1024), (resource.RLIMIT_CORE, 0)):
            resource.setrlimit(kind, (maximum, maximum))

    def bounded_popen(*args, **kwargs):
        need("preexec_fn" not in kwargs, "unexpected_child_launcher")
        return original_popen(*args, preexec_fn=child_limits, **kwargs)

    _c.subprocess = types.SimpleNamespace(Popen=bounded_popen, PIPE=_c.subprocess.PIPE,
        DEVNULL=_c.subprocess.DEVNULL, TimeoutExpired=_c.subprocess.TimeoutExpired)
    _c.report = {"schemaVersion": "quantopsy.embedding-study.v2", "sourceCommit": _c.source_commit,
                 "attempt": 1, "fixtureTestsExecutedThisInvocation": False,
                 "canonicalBaseCommit": BASE_COMMIT, "packetStatus": "failed", "operationalCompleted": False,
                 "suppliedRecoveryRecordsPass": False, "globalBehavioralRecoveryAccepted": False,
                 "startedAtUtc": _c.utc(), "commands": [], "guards": [], "sourceFiles": [], "downloads": [],
                 "bindings": [], "inspections": [], "errors": [], "storageSamples": [], "generatedFiles": [],
                 "scope": {"fixtureCasesCarried": 74, "fixtureExecution": "published_v3_evidence_only",
                           "realModels": _c.FIXTURES, "taskCount": 8,
                           "modelRoles": ["reference", "quantized", "candidate-v2", "candidate-v3"],
                           "manualSelectedTensors": {"v2": 42, "v3": 43},
                           "protocol": PROTOCOL, "protocolSha256": PROTOCOL_SHA256,
                           "inference": "teacher_forced_target_tokens_cpu", "sampling": False},
                 "limits": {"jobMinutes": 20, "workSeconds": 900, "cleanupSeconds": 90, "marginSeconds": 30,
                            "sourceArchiveBytes": MAX_ARCHIVE, "extractedBytes": MAX_EXTRACTED, "buildBytes": MAX_BUILD,
                            "buildJobs": 2, "storageBytes": MAX_STORAGE, "files": MAX_FILES, "directories": MAX_DIRECTORIES,
                            "childAddressSpaceBytes": 2 * 1024**3, "childCpuSeconds": 900, "childOpenFiles": 1024,
                            "streamBytes": _c.STREAM_LIMIT, "reportBytes": _c.REPORT_LIMIT,
                            "selectedElements": MAX_ELEMENTS, "maxAddedWeightBytes": MAX_ADDED_BYTES},
                 "proofLimits": ["Eight authored continuation tasks measure conditional target NLL and teacher-forced next-token accuracy, not general semantic quality or global behavior.",
                                 "A completed negative experiment remains operationally completed; command success does not imply recovery.",
                                 "The downloaded complete archive hash is observed; full native Git tree reconstruction separately binds its source files and one unfetched pinned gitlink to the reviewed commit tree. Model cryptographic lineage remains unproved.",
                                 "Application runtime remains Python standard library only; the fixed C++ scorer and pinned upstream build exist only in this owned CI packet.",
                                 "The 74 distinct native fixture checks are carried with exact product/shared-source and two-literal proofs; no fixture case is repeated here.",
                                 "No model, tokenizer payload or build output is distributed; raw per-token stdout and all normal command closures remain in this bounded receipt."]}


def main():
    global _data
    completed = False
    try:
        install_support()
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, lambda *_: setattr(_c, "interrupted", True))
        preflight()
        fixture_checks()
        _data = load_fixed(_c.repo / DATA_HELPER, "quantopsy_fixed_behavioral_data",
                           next(row["sha256"] for row in _c.source_vector if row["path"] == DATA_HELPER))
        protocol = _data.protocol_contract(_c.repo)
        need(_c.sha(_c.read(_c.repo / PROTOCOL)) == PROTOCOL_SHA256
             and protocol["decision"]["complete_role_runs_required"] == 4
             and protocol["decision"]["comparisons_required"] == 2, "predeclared_execution_contract_changed")
        _c.report["protocolFrozenBeforeDownloads"] = {"sha256": PROTOCOL_SHA256, "utc": _c.utc()}
        for fixture in _c.FIXTURES:
            _c.download(fixture)
        need(len(_c.report["downloads"]) == 2 and sum(row["receivedBytes"] for row in _c.report["downloads"]) == _c.DOWNLOAD_TOTAL
             and all(row["complete"] and row["bodyEOFObserved"] and row["responseClosed"] and row["fileDescriptorClosed"]
                     and not row["resourceErrors"] for row in _c.report["downloads"]), "model_downloads_incomplete")
        pair = _c.binding_phase("pair")
        _c.owned_write("output/provenance.json", _c.canonical(_c.PROVENANCE))
        candidate_inventories = {}
        for profile in ("v2", "v3"):
            selected, target = selection_rows(pair, profile)
            candidate_inventories[profile] = candidate_bundle(pair, selected, target, profile)
        source_guard("after_candidates")
        download_source()
        extract_source()
        build_engine()
        suite_raw, config_raw, models = freeze_measurements(pair, candidate_inventories)
        inference(suite_raw, config_raw, models)
        completed = True
    except Exception as error:
        if _c is None:
            raise
        _c.report["errors"].append(_c.finite(error))
        frame = error.__traceback__
        while frame is not None and frame.tb_next is not None:
            frame = frame.tb_next
        _c.report["errorLocation"] = None if frame is None else {"file": Path(frame.tb_frame.f_code.co_filename).name[:100], "line": frame.tb_lineno}
        _c.report["errorType"] = type(error).__name__[:80]
    finally:
        if _c is not None:
            cleanup()
            if _c.interrupted:
                _c.report["errors"].append("interrupted")
            lifecycle = all(row.get("normallyClosed") is True for row in _c.report["commands"])
            zero = all(row["exitCode"] == 0 for row in _c.report["commands"])
            operational = bool(completed and not _c.report["errors"] and not _c.interrupted and lifecycle and zero
                               and _c.report.get("cleanup", {}).get("passed") is True
                               and _c.report.get("scratchCleanup") == "passed" and all(row["passed"] for row in _c.report["guards"])
                               and _c.report.get("fixture", {}).get("accepted") is True
                               and _c.work_end is not None and time.monotonic() <= _c.work_end + 90)
            _c.report.update({"operationalCompleted": operational, "packetStatus": "completed" if operational else "failed",
                              "completedAtUtc": _c.utc(), "processCleanup": {"allRowsNormallyClosed": lifecycle, "allCommandExitsZero": zero,
                              "signalsSent": any(row["termSent"] or row["killSent"] for row in _c.report["commands"])}})
            if not operational:
                _c.report["suppliedRecoveryRecordsPass"] = False
            output = json.dumps(_c.report, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
            need(len(output) <= _c.REPORT_LIMIT, "report_bound")
            signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
            signal.setitimer(signal.ITIMER_REAL, 2)
            sys.stdout.buffer.write(PREFIX + output + b"\n")
            sys.stdout.buffer.flush()
            signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if _c.report["operationalCompleted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
