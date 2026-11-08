#!/usr/bin/env python3
"""One bounded public-CI task-NLL experiment; no local execution or inference."""

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


BASE_COMMIT = "f2b3c812976dafcf8507713c8bb24ccdd4f63979"
REPOSITORY = "alvarovillalbaa/quantopsy"
BRANCH = "refs/heads/qa/quantopsy-behavior-20261108-2"
ACTIVE = ".github/workflows/behavioral-inference.yml"
# The final prepared inert registration stub is filled by the publishing owner.
# An absent pin is deliberately an execution refusal.
INERT_WORKFLOW_SHA256 = "9cb24e746af0585cdddb321e8c0f94198b6ba629ebdd43b897b0d10fc8b29060"
CONTROLLER = "tests/scripts/run-behavioral-inference.py"
SUPPORT = "tests/scripts/run-public-gguf-v2-compatibility.py"
SUPPORT_SHA256 = "5161f1b9f89f6b8534f2e036dce06850082e4579e427417fa1ba0411cce9b99d"
DATA_HELPER = "tests/helpers/behavioral_inference.py"
SCORER = "tests/helpers/behavioral_scorer.cpp"
CMAKE = "tests/helpers/behavioral_engine/CMakeLists.txt"
LLAMA_REVISION = "74d4f5b041ad837153b0e90fc864b8290e01d8d5"
LLAMA_TREE = "9ae6f9dd0984b780c942ccf510e9cd5fcee9e724"
ARCHIVE_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/" + LLAMA_REVISION
ARCHIVE_ROOT = "llama.cpp-" + LLAMA_REVISION
MODIFIED_PRIOR = {CONTROLLER, "AGENTS.md", "README.md", "tests/README.md", "docs/proposals/behavioral-inference-v1.md"}
PRIOR_RECEIPT = "docs/results/2026/11-08/behavioral-inference-attempt-1.json"
PRIOR_RECEIPT_SHA256 = "e82d1e384bc4626ca210403580800ed3b7cc0f7ac30ecdd9887c1e8990ff031a"
ADDED_PATHS = {PRIOR_RECEIPT, "docs/results/2026/11-08/behavioral-inference-attempt-1.log",
               "docs/results/2026/11-08/behavioral-inference-attempt-1.md"}
MAX_ARCHIVE = 64 * 1024**2
MAX_EXTRACTED = 512 * 1024**2
MAX_BUILD = 2 * 1024**3
MAX_STORAGE = 3 * 1024**3
MAX_FILES = 30_000
MAX_DIRECTORIES = 8_000
MAX_ELEMENTS = 5_971_968
ADDED_WEIGHT_BYTES = 20_528_640
MAX_ADDED_BYTES = 64 * 1024**2
RESERVE_BYTES = 1024**3
EXPECTED_BEHAVIOR_CASES = 35
METRICS = [
    {"id": "mean_target_nll", "direction": "lower", "min_quantization_loss": "0.005",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.02", "max_quantized_regression": "0.01"},
    {"id": "target_token_accuracy", "direction": "higher", "min_quantization_loss": "0.02",
     "min_recovery_fraction": "0.5", "max_reference_regression": "0.03", "max_quantized_regression": "0.02"},
]
PREFIX = b"QUANTOPSY_BEHAVIORAL_INFERENCE_RESULT="
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
    if os.path.lexists(_c.root / "output/candidate"):
        admit_directory("output/candidate")
    for name in ("output/repair-plan-v2.json", "output/candidate/candidate.gguf", "output/candidate/plan.json",
                 "output/candidate/repair.json", "output/comparison.json"):
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
    need(len(INERT_WORKFLOW_SHA256) == 64 and _c.sha(_c.read(_c.repo / ACTIVE)) == INERT_WORKFLOW_SHA256,
         "inert_workflow_pin_missing_or_changed")
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
         and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
         and os.environ.get("SOURCE_COMMIT") == _c.source_commit
         and os.environ.get("QUANTOPSY_ARTIFACT_PUBLIC_REPOSITORY") == "true", "approved_public_ci_required")
    event = _c.strict_json(_c.read(os.environ["GITHUB_EVENT_PATH"], 262144))
    need(("ref" not in event or event["ref"] in (BRANCH, BRANCH.removeprefix("refs/heads/")))
         and event.get("repository", {}).get("private") is False
         and event.get("repository", {}).get("full_name") == REPOSITORY, "approved_dispatch_event_required")
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
    """Bind closed checks from attempt 1; execute no test or application code."""
    raw = _c.read(_c.repo / "tests/test_behavior.py", 128 * 1024)
    tree = ast.parse(raw)
    names = [node.name for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "BehaviorComparisonTier2Tests"
             for node in item.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    need(len(names) == len(set(names)) == EXPECTED_BEHAVIOR_CASES, "new_fixture_identity_count")
    cli_ids = [name for name in _c.FIXTURE_IDS if name.startswith("tests.test_cli.")]
    expected = sorted(["tests.test_behavior.BehaviorComparisonTier2Tests." + name for name in names] + cli_ids)
    need(len(cli_ids) == 16 and len(expected) == EXPECTED_BEHAVIOR_CASES + 16, "fixture_regression_count")
    raw = _c.read(_c.repo / PRIOR_RECEIPT)
    need(_c.sha(raw) == PRIOR_RECEIPT_SHA256, "prior_fixture_receipt_hash_mismatch")
    prior = _c.strict_json(raw)
    need(prior["sourceCommit"] == BASE_COMMIT
         and prior["canonicalBaseCommit"] == "87f94dbe5af72b8492e53a7096147d617c810e0d"
         and prior["fixture"]["accepted"] is True
         and prior["fixture"]["modules"] == ["tests.test_behavior", "tests.test_cli"]
         and prior["fixture"]["expected"] == 51, "prior_fixture_receipt_identity")
    value = prior["fixture"]["actual"]
    commands = [row for row in prior["commands"] if row["label"] == prior["fixture"]["originalCommand"]]
    need(len(commands) == 1 and commands[0]["exitCode"] == 0 and commands[0]["normallyClosed"] is True
         and all(commands[0][stream]["complete"] is True and commands[0][stream]["retained"] is True
                 for stream in ("stdout", "stderr")), "prior_fixture_command_incomplete")
    previous = {row["path"]: row for row in prior["sourceFiles"]}
    current = {row["path"]: row for row in _c.source_vector}
    fixed = sorted(name for name in previous if (name.startswith("quantopsy/") or name.startswith("tests/"))
                   and name not in (CONTROLLER, "tests/README.md"))
    need(fixed and all(name in current and current[name] == previous[name] for name in fixed),
         "prior_fixture_application_or_test_input_changed")
    need({name for name in current if name.startswith(("quantopsy/", "tests/")) and name not in (CONTROLLER, "tests/README.md")}
         == set(fixed), "prior_fixture_input_set_changed")
    _c.report["fixture"] = {"modules": ["tests.test_behavior", "tests.test_cli"], "expected": len(expected),
                             "actual": value, "accepted": False, "executed": False,
                             "evidence": {"file": PRIOR_RECEIPT, "sha256": PRIOR_RECEIPT_SHA256, "sourceCommit": BASE_COMMIT,
                                          "originalCommand": commands[0]["label"], "matchedSourceInputs": fixed,
                                          "priorPython": prior["python"]}}
    need(type(value["collected"]) is int and type(value["testsRun"]) is int and value["collected"] == value["testsRun"] == len(expected)
         and value["identities"] == expected and all(type(value[key]) is int and value[key] == 0 for key in
         ("failures", "errors", "skips", "expectedFailures", "unexpectedSuccesses")), "fixture_checks_failed")
    _c.report["fixture"]["accepted"] = True
    need(not any((_c.root / "tmp").iterdir()), "fixture_temporary_residue_preserved")


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
    candidate=m.HeldFile(root,'output/candidate/candidate.gguf',67108864,rows,opened)
    inventory=m.observe_payloads(observer,candidate)
finally:
    for held in opened:held.verify()
    for held in reversed(opened):held.close()
m.root_guard(root,os.environ['QUANTOPSY_ARTIFACT_NONCE'],original,marker)
assert all(row['unchanged'] and row['closed'] for row in rows)
print('QUANTOPSY_BEHAVIOR_CANDIDATE_INVENTORY='+json.dumps({'inventory':inventory,'files':rows},sort_keys=True,separators=(',',':')))
'''


def selection_rows(pair):
    reference = {row["name"]: row for row in pair["inputs"][0]["inventory"]["tensors"]}
    quantized = {row["name"]: row for row in pair["inputs"][1]["inventory"]["tensors"]}
    expected_names = ["blk." + str(layer) + "." + name + ".weight" for layer in range(6)
                      for name in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")]
    need(list(_data.SELECTED_TENSORS) == expected_names, "manual_selection_changed")
    rows = []
    for name in expected_names:
        a, b = reference[name], quantized[name]
        need(a["ggmlType"] == 0 and b["ggmlType"] == 2 and a["dimensions"] == b["dimensions"] and len(a["dimensions"]) == 2,
             "manual_selection_type_or_shape")
        rows.append({"name": name, "dimensions": a["dimensions"], "elements": a["elements"], "reference_kind": 0,
                     "quantized_kind": 2, "reference_sha256": a["payloadSha256"], "quantized_sha256": b["payloadSha256"],
                     "reference_bytes": a["payloadBytes"], "quantized_bytes": b["payloadBytes"]})
    need(len(rows) == 42 and sum(row["elements"] for row in rows) == MAX_ELEMENTS
         and sum(row["reference_bytes"] - row["quantized_bytes"] for row in rows) == ADDED_WEIGHT_BYTES, "manual_selection_work_or_growth")
    _c.report["manualSelection"] = {"names": expected_names, "elements": MAX_ELEMENTS, "addedWeightBytes": ADDED_WEIGHT_BYTES,
                                    "frozenBeforeCandidateOutputs": True, "automaticSelection": False}
    return rows


def candidate_bundle(pair, selected):
    _c.owned_write("output/provenance.json", _c.canonical(_c.PROVENANCE))
    args = ["plan", _c.root / "models" / _c.FIXTURES[0]["name"], _c.root / "models" / _c.FIXTURES[1]["name"], "--profile", "v2"]
    for row in selected:
        args.extend(["--tensor", row["name"]])
    args.extend(["--provenance", _c.root / "output/provenance.json", "--max-added-bytes", MAX_ADDED_BYTES,
                 "--max-elements", MAX_ELEMENTS, "--output", _c.root / "output/repair-plan-v2.json"])
    _c.product_command("plan_42_explicit_matrices", args, outputs=True)
    plan, plan_raw = _c.read_owned_json("output/repair-plan-v2.json")
    expected = {"schema_version": 2, "kind": "quantopsy.repair-plan", "architecture": "llama", "provenance": _c.PROVENANCE,
                "reference": {"name": _c.FIXTURES[0]["name"], "sha256": _c.FIXTURES[0]["sha256"], "size": _c.FIXTURES[0]["bytes"]},
                "quantized": {"name": _c.FIXTURES[1]["name"], "sha256": _c.FIXTURES[1]["sha256"], "size": _c.FIXTURES[1]["bytes"]},
                "selected": selected, "added_weight_bytes": ADDED_WEIGHT_BYTES, "max_added_bytes": MAX_ADDED_BYTES,
                "max_elements": MAX_ELEMENTS, "lineage": "operator_declared_compatibility_checked", "behavioral_validation": "not_run"}
    need(plan == expected and plan_raw == _c.canonical(expected), "plan_does_not_match_frozen_42_matrix_intent")
    cli, cli_raw = _c.product_command("repair_42_explicit_matrices", ["repair", _c.root / "models" / _c.FIXTURES[0]["name"],
        _c.root / "models" / _c.FIXTURES[1]["name"], "--profile", "v2", "--plan", _c.root / "output/repair-plan-v2.json",
        "--output-dir", _c.root / "output/candidate", "--reserve-bytes", RESERVE_BYTES], outputs=True)
    manifest, manifest_raw = _c.read_owned_json("output/candidate/repair.json")
    bundled, bundled_raw = _c.read_owned_json("output/candidate/plan.json")
    need(cli == manifest and cli_raw == manifest_raw and bundled == plan and bundled_raw == plan_raw, "manifest_plan_originals_differ")
    current = fingerprint(_c.root / "output/candidate/candidate.gguf")
    need(current["bytes"] <= 64 * 1024**2, "candidate_file_bound")
    out, err, command = _c.run("independent_57_payload_candidate_inventory", [sys.executable, "-I", "-B", "-S", "-c", OBSERVE_CANDIDATE], 100)
    prefix = b"QUANTOPSY_BEHAVIOR_CANDIDATE_INVENTORY="
    need(err == b"" and out.startswith(prefix) and out.endswith(b"\n") and out.count(prefix) == 1, "candidate_inventory_original_marker")
    value = _c.strict_json(out[len(prefix):-1])
    inventory = value["inventory"]
    _c.validate_inventory(_c.root / current["relativePath"], current["bytes"], inventory)
    for row in value["files"]:
        original = fingerprint(_c.root / row["relativePath"])
        fields = {key: original[key] for key in ("device", "inode", "uid", "mode", "bytes", "mtimeNs", "ctimeNs")}
        need(row["before"] == row["after"] == fields and row["sha256Before"] == row["sha256After"] == original["sha256"]
             and row["closed"] is True and row["unchanged"] is True, "candidate_observer_file_identity")
    reference = {row["name"]: row for row in pair["inputs"][0]["inventory"]["tensors"]}
    q_inventory = pair["inputs"][1]["inventory"]
    quantized = {row["name"]: row for row in q_inventory["tensors"]}
    need([row["name"] for row in inventory["tensors"]] == list(quantized), "candidate_tensor_order")
    expected_metadata = [row for row in q_inventory["metadata"] if row["key"] != "general.file_type"]
    need([row["key"] for row in inventory["metadata"]] == [row["key"] for row in expected_metadata]
         and all(_c.metadata_signature(a) == _c.metadata_signature(b) for a, b in zip(inventory["metadata"], expected_metadata)),
         "candidate_metadata_preservation")
    payloads = []
    selected_names = {row["name"] for row in selected}
    for actual in inventory["tensors"]:
        name = actual["name"]
        source = reference[name] if name in selected_names else quantized[name]
        need((actual["dimensions"], actual["ggmlType"], actual["payloadBytes"], actual["payloadSha256"])
             == (source["dimensions"], source["ggmlType"], source["payloadBytes"], source["payloadSha256"]), "candidate_prescribed_payload_mismatch")
        payloads.append({"name": name, "source": "reference" if name in selected_names else "quantized",
                         "bytes": actual["payloadBytes"], "sha256": actual["payloadSha256"]})
    expected_manifest = {key: value for key, value in plan.items() if key not in ("kind", "schema_version")}
    expected_manifest.update({"schema_version": 2, "kind": "quantopsy.repair-result", "plan_sha256": _c.sha(plan_raw),
                              "output": {"name": "candidate.gguf", "sha256": current["sha256"], "size": current["bytes"]},
                              "reserve_bytes": RESERVE_BYTES, "metadata_omissions": ["general.file_type"] if len(expected_metadata) != len(q_inventory["metadata"]) else [],
                              "verification": {"container_reparsed": True, "metadata_preserved": True, "tensor_payloads_verified": True, "source_hashes_verified": True}})
    need(manifest == expected_manifest and manifest_raw == _c.canonical(expected_manifest)
         and len(payloads) == 57 and sum(row["source"] == "reference" for row in payloads) == 42,
         "candidate_manifest_or_binding_count")
    _c.report["candidate"] = {"plan": plan, "planSha256": _c.sha(plan_raw), "manifest": manifest,
                              "manifestSha256": _c.sha(manifest_raw), "independentPayloads": payloads,
                              "inventoryOriginalCommand": command["label"], "candidateIdentity": current}
    return inventory


def tokenizer_hash(inventory):
    fields = ("key", "kind", "arrayKind", "count", "encodedBytes", "encodedSha256")
    rows = [{key: row[key] for key in fields} for row in inventory["metadata"] if row["key"].startswith("tokenizer.")]
    need(rows, "tokenizer_identity_missing")
    return _c.sha(_data.canonical(sorted(rows, key=lambda row: row["key"])))


def freeze_measurements(pair, candidate_inventory):
    tokenizers = [tokenizer_hash(row["inventory"]) for row in pair["inputs"]] + [tokenizer_hash(candidate_inventory)]
    need(len(set(tokenizers)) == 1, "tokenizer_triplet_mismatch")
    manifest = _data.engine_manifest(_c.repo)
    manifest_raw = _data.canonical(manifest)
    config = _data.engine_config(manifest_raw, tokenizers[0])
    config_raw = _data.canonical(config)
    suite = _data.frozen_suite(config_raw)
    need(suite["metrics"] == METRICS and len(suite["tasks"]) == 8, "frozen_suite_protocol_changed")
    suite_raw = _data.canonical(suite)
    models = {"schema_version": 1, "kind": "quantopsy.behavior-models"}
    for role, relative in zip(("reference", "quantized", "candidate"),
                              ("models/" + _c.FIXTURES[0]["name"], "models/" + _c.FIXTURES[1]["name"], "output/candidate/candidate.gguf")):
        row = fingerprint(_c.root / relative)
        models[role] = {"name": Path(relative).name, "size": row["bytes"], "sha256": row["sha256"]}
    records = {"output/engine-manifest.json": manifest_raw, "output/config.json": config_raw,
               "output/suite.json": suite_raw, "output/models.json": _c.canonical(models),
               "output/tasks.tsv": _data.encode_tasks(suite)}
    for relative, raw in records.items():
        _c.owned_write(relative, raw)
        _frozen[relative] = fingerprint(_c.root / relative)
    _frozen["build/quantopsy-behavioral-scorer"] = _binary
    originals = {relative: _c.capture(raw, len(raw), _c.sha(raw), True) for relative, raw in records.items()}
    need(all(row["complete"] and row["retained"] for row in originals.values()), "frozen_original_retention_failed")
    _c.report["measurementFreeze"] = {"beforeAnyInference": True, "utc": _c.utc(), "files": list(_frozen.values()),
                                       "originals": originals,
                                       "metrics": METRICS, "taskCount": 8, "tokenizerSha256": tokenizers[0],
                                       "engineManifest": manifest, "configuration": config,
                                       "sourceVectorSha256": _c.sha(_c.canonical(_c.source_vector))}
    return suite_raw, config_raw, models


def frozen_guard(label):
    _c.verify_inputs(label)
    for relative, previous in _frozen.items():
        need(fingerprint(_c.root / relative) == previous, "frozen_measurement_input_changed")
    _c.verify_generated(label)


def inference(suite_raw, config_raw, models):
    details = {}
    for role, relative in zip(("reference", "quantized", "candidate"),
                              ("models/" + _c.FIXTURES[0]["name"], "models/" + _c.FIXTURES[1]["name"], "output/candidate/candidate.gguf")):
        frozen_guard("before_inference_" + role)
        out, err, command = _c.run("real_cpu_forward_" + role, [_c.root / "build/quantopsy-behavioral-scorer", _c.root / relative,
                                  _c.root / "output/tasks.tsv"], 90, cwd=_c.root / "output")
        collected = _data.collect_run(out, role, suite_raw, config_raw, models[role])
        need(set(collected) == {"run", "details"}, "scorer_collection_shape")
        details[role] = collected["details"]
        run_raw = _c.canonical(collected["run"])
        _c.owned_write("output/" + role + "-run.json", run_raw)
        _c.owned_write("output/" + role + "-token-details.json", _c.canonical(collected["details"]))
        _c.report.setdefault("inferenceRuns", []).append({"role": role, "originalCommand": command["label"],
                                                          "run": collected["run"], "runOriginal": _c.capture(run_raw, len(run_raw), _c.sha(run_raw), True),
                                                          "detailsSha256": _c.sha(_c.canonical(collected["details"])),
                                                          "stderrOriginalRetained": command["stderr"]["complete"]})
        frozen_guard("after_inference_" + role)
    _data.compare_tokenization(details)
    _c.report["tokenizationAcrossRoles"] = "exact_prompt_and_target_token_ids_verified"
    args = ["compare-behavior", "--suite", _c.root / "output/suite.json", "--config", _c.root / "output/config.json",
            "--models", _c.root / "output/models.json", "--repair-manifest", _c.root / "output/candidate/repair.json",
            "--reference-run", _c.root / "output/reference-run.json", "--quantized-run", _c.root / "output/quantized-run.json",
            "--candidate-run", _c.root / "output/candidate-run.json", "--output", _c.root / "output/comparison.json"]
    _c.product_command("compare_frozen_real_inference_records", args, outputs=True)
    comparison, raw = _c.read_owned_json("output/comparison.json")
    _c.report["comparison"] = {"value": comparison, "bytes": len(raw), "sha256": _c.sha(raw)}
    need(comparison["acceptance"]["complete_runs"] is True
         and type(comparison["acceptance"]["supplied_recovery_records_pass"]) is bool,
         "comparison_did_not_cover_all_completed_runs")
    _c.report["suppliedRecoveryRecordsPass"] = comparison["acceptance"]["supplied_recovery_records_pass"]
    frozen_guard("after_comparison")
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
    _c.report = {"schemaVersion": "quantopsy.behavioral-inference.v1", "sourceCommit": _c.source_commit,
                 "attempt": 2, "fixtureTestsExecutedThisInvocation": False,
                 "canonicalBaseCommit": BASE_COMMIT, "packetStatus": "failed", "operationalCompleted": False,
                 "suppliedRecoveryRecordsPass": False, "globalBehavioralRecoveryAccepted": False,
                 "startedAtUtc": _c.utc(), "commands": [], "guards": [], "sourceFiles": [], "downloads": [],
                 "bindings": [], "inspections": [], "errors": [], "storageSamples": [], "generatedFiles": [],
                 "scope": {"fixtureModules": ["tests.test_behavior", "tests.test_cli"], "legacyCases": 16,
                           "fixtureExecution": "retained_attempt_1_evidence_only",
                           "realModels": _c.FIXTURES, "taskCount": 8, "modelRoles": ["reference", "quantized", "candidate"],
                           "manualSelectedMatrices": 42, "inference": "teacher_forced_target_tokens_cpu", "sampling": False},
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
                                 "The 51 unchanged application/CLI checks are retained attempt-1 evidence, not tests executed by this invocation; fresh inference and teardown remain required.",
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
        for fixture in _c.FIXTURES:
            _c.download(fixture)
        need(len(_c.report["downloads"]) == 2 and sum(row["receivedBytes"] for row in _c.report["downloads"]) == _c.DOWNLOAD_TOTAL
             and all(row["complete"] and row["bodyEOFObserved"] and row["responseClosed"] and row["fileDescriptorClosed"]
                     and not row["resourceErrors"] for row in _c.report["downloads"]), "model_downloads_incomplete")
        pair = _c.binding_phase("pair")
        selected = selection_rows(pair)
        candidate_inventory = candidate_bundle(pair, selected)
        download_source()
        extract_source()
        build_engine()
        suite_raw, config_raw, models = freeze_measurements(pair, candidate_inventory)
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
