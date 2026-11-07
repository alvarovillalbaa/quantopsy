#!/usr/bin/env python3
"""Bounded public GGUF observations; no inference, conversion or fixture rerun."""
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

BASE_COMMIT = "c554e86f38c1a64cbb73dd02ff99bf9129f6c76c"
REPOSITORY = "alvarovillalbaa/quantopsy"
BRANCH = "refs/heads/qa/quantopsy-artifacts-20261107-1"
ACTIVE = ".github/workflows/public-gguf-observation-1.yml"
HELPER = "tests/helpers/public_gguf_inventory.py"
HELPER_SHA256 = "e1a51e561fc9ba2ab0a2b4e3604575accd37d2b560eebd7cda62772bda32c1f5"
AGENTS_SHA256 = "211d764b4045d7297fc392a4b41e824e5f865aebed2a7d3a52aa78127183b13c"
ADDED = sorted([HELPER, "tests/README.md", "tests/scripts/run-public-gguf-observation.py",
                "docs/proposals/public-gguf-observation.md", "docs/proposals/public-gguf-observation.workflow.yml.example"])
FIXTURES = [
    {"id": "stories260k-f32", "helperFixture": "stories260k-f32", "name": "stories260K-f32.gguf", "bytes": 1185376,
     "sha256": "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d",
     "url": "https://huggingface.co/ggml-org/test-model-stories260K/resolve/479896ec924af6d40fd419ab8f4d1eb2101de00d/stories260K-f32.gguf"},
    {"id": "stories15m-reference", "helperFixture": "stories15m-reference", "name": "stories15M.gguf", "bytes": 98357920,
     "sha256": "61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a",
     "url": "https://huggingface.co/ggml-org/models-moved/resolve/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M.gguf"},
    {"id": "stories15m-q4_0", "helperFixture": "stories15m-q4_0", "name": "stories15M-q4_0.gguf", "bytes": 19077344,
     "sha256": "6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04",
     "url": "https://huggingface.co/ggml-org/models-moved/resolve/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M-q4_0.gguf"},
]
DOWNLOAD_TOTAL = 118620640
STORAGE_LIMIT = 512 * 1024**2
STREAM_LIMIT, ZIP_LIMIT, ALL_ZIP_LIMIT, REPORT_LIMIT = 1024**2, 256 * 1024, 1024**2, 8 * 1024**2
SELECTED_TENSOR = "blk.0.attn_q.weight"
CDN_HOSTS = {"huggingface.co", "cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co", "cas-bridge.xethub.hf.co",
             "aws.cdn.hf.co", "us.aws.cdn.hf.co", "us-east-1.aws.cdn.hf.co", "us-west-2.aws.cdn.hf.co",
             "eu-west-3.aws.cdn.hf.co", "ap-southeast-1.aws.cdn.hf.co", "us.gcp.cdn.hf.co",
             "us-east1.us.gcp.cdn.hf.co", "us-central1.us.gcp.cdn.hf.co", "us-west4.us.gcp.cdn.hf.co",
             "europe-west4.us.gcp.cdn.hf.co", "asia-southeast1.us.gcp.cdn.hf.co"}
sha = lambda data: hashlib.sha256(data).hexdigest()
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
source_commit = sys.argv[2] if len(sys.argv) == 3 and sys.argv[1] == "--source-commit" else None
report = {
    "schemaVersion": "quantopsy.public-gguf-observation.v1", "sourceCommit": source_commit,
    "canonicalBaseCommit": BASE_COMMIT, "packetStatus": "failed", "diagnosticComplete": False,
    "productAccepted": False, "startedAtUtc": utc(), "commands": [], "guards": [], "sourceFiles": [],
    "downloads": [], "observations": [], "errors": [], "storageSamples": [],
    "pairRestorationAdmission": {"accepted": False, "status": "not_evaluated", "selectedTensor": SELECTED_TENSOR},
    "scope": {"tier": "tier2", "layer": "operational-cli-file-integration", "files": FIXTURES,
              "inference": False, "modelCode": False, "dependencies": False, "closed80Repeated": False,
              "diagnose": False, "probe": False, "plan": False, "repair": False, "conversion": False},
    "limits": {"jobMinutes": 15, "workSeconds": 600, "cleanupSeconds": 90, "marginSeconds": 30,
               "downloadSeconds": 120, "redirectsPerFile": 5, "downloadBytesTotal": DOWNLOAD_TOTAL,
               "logicalStorageBytes": STORAGE_LIMIT, "streamBytes": STREAM_LIMIT, "streamGzipBytes": ZIP_LIMIT,
               "allStreamsGzipBytes": ALL_ZIP_LIMIT, "reportBytes": REPORT_LIMIT, "stdoutSeconds": 2},
    "proofLimits": [
        "Diagnostic completion is not per-file container admission, restoration-pair admission or recovered behavior.",
        "No model/tokenizer code, inference, conversion, repair, provider account or closed authored-fixture check runs.",
        "Only public pinned inputs are downloaded; original outputs contain bounded structural descriptors and metadata hashes, not weights/tokenizer arrays.",
        "Signed redirect queries stay memory-only. Literal HTTPS host admission and public DNS checks are not a host-wide network attestation.",
        "UID/PID/starttime/group and owned-root inventories qualify cleanup, not every host process. External/kernel termination can prevent verified cleanup.",
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

def canonical_base():
    global base_tree, base_agents
    stdout, _, _ = run("canonical_parent", ["git", "rev-parse", "HEAD^"], 10)
    need(stdout.strip().decode() == BASE_COMMIT, "canonical_parent_changed")
    stdout, _, _ = run("canonical_tree", ["git", "ls-tree", "-r", "-l", "-z", BASE_COMMIT], 10)
    base_tree = parse_git(stdout, sizes=True)
    need(len(base_tree) == 31 and sum(x["bytes"] for x in base_tree.values()) == 242315
         and not any(n in base_tree for n in ADDED), "canonical_baseline_changed")
    base_agents, _, _ = run("canonical_agents", ["git", "cat-file", "blob", base_tree["AGENTS.md"]["gitBlob"]], 10)
    report["canonicalBaseline"] = {"commit": BASE_COMMIT, "blobs": 31, "bytes": 242315,
                                   "onlyModifiedPriorPaths": ["AGENTS.md"], "onlyAddedPaths": ADDED}

def guard(label, cleanup=False, post_scratch=False):
    global source_vector
    item = {"label": label, "passed": False}
    report["guards"].append(item)
    options = {"cleanup": cleanup, "post_scratch": post_scratch}
    out, _, _ = run(label + "_head", ["git", "rev-parse", "HEAD"], 10, **options)
    need(out.strip().decode() == source_commit and not (repo / ACTIVE).exists(), "source_head_or_workflow_changed")
    tree, _, _ = run(label + "_tree", ["git", "ls-tree", "-r", "-z", "HEAD"], 10, **options)
    index, _, _ = run(label + "_index", ["git", "ls-files", "--stage", "-z"], 10, **options)
    tree, index = parse_git(tree), parse_git(index, index=True)
    need(base_tree is not None and len(tree) == len(index) == 36, "source_path_count")
    need(sorted(set(tree) - set(base_tree)) == ADDED, "source_added_paths")
    vector = []
    for name, previous in base_tree.items():
        need(tree[name]["mode"] == previous["mode"], "prior_source_mode_changed")
        if name == "AGENTS.md":
            current = read(repo / name)
            need(sha(current) == AGENTS_SHA256 and len(current) > len(base_agents)
                 and current.startswith(base_agents), "scope_document_changed")
        else:
            need(tree[name]["gitBlob"] == previous["gitBlob"], "prior_source_blob_changed")
    for name, record in tree.items():
        actual = file_vector(name)
        need(index[name] == record and actual["mode"] == record["mode"] and actual["gitBlob"] == record["gitBlob"], "source_index_or_body_changed")
        vector.append(actual)
    for suffix, args in (("status", ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]),
                         ("ignored", ["git", "status", "--porcelain=v1", "-z", "--ignored=matching", "--untracked-files=normal"])):
        out, _, _ = run(label + "_" + suffix, args, 10, **options)
        need(out == b"", "source_generated_or_dirty_output")
    run(label + "_diff", ["git", "-c", "core.filemode=true", "diff", "--no-ext-diff", "--no-textconv", "--quiet", "--exit-code", "HEAD", "--"], 10, **options)
    if source_vector is None:
        source_vector = vector
        report["sourceFiles"] = vector
    need(vector == source_vector, "source_vector_changed")
    item.update({"passed": True, "blobs": len(vector), "bytes": sum(x["bytes"] for x in vector),
                 "vectorSha256": sha(json.dumps(vector, ensure_ascii=False, separators=(",", ":")).encode()),
                 "fullGitIndexNativeBindings": True, "allPriorApplicationInputsUnchanged": True})

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


SCALAR_KEYS = {"general.architecture", "general.name", "general.alignment", "general.file_type", "general.quantization_version",
               "llama.context_length", "llama.embedding_length", "llama.feed_forward_length", "llama.block_count",
               "llama.attention.head_count", "llama.attention.head_count_kv", "llama.rope.dimension_count",
               "llama.attention.layer_norm_rms_epsilon"}

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

def validate_inventory(fixture, value):
    file = root / "models" / fixture["name"]
    need(value.get("schemaVersion") == 1 and value.get("sourceCommit") == source_commit and value.get("status") == "observed"
         and value.get("fixture") == fixture["helperFixture"] and value.get("errors") == []
         and value.get("applicationAdmission") == value.get("pairRestorationAdmission") == "not_evaluated", "inventory_observation_incomplete")
    need(value.get("expectedArtifact") == {"bytes": fixture["bytes"], "sha256": fixture["sha256"],
         "revision": fixture["url"].split("/resolve/")[1].split("/")[0], "url": fixture["url"]}, "inventory_fixture_binding")
    expected_identity = {k: tracked_files[fixture["id"]][k] for k in
                         ("device", "inode", "uid", "mode", "bytes", "mtimeNs", "ctimeNs")}
    need(value["input"]["before"] == value["input"]["after"] == expected_identity
         and value["input"]["sha256Before"] == value["input"]["sha256After"] == fixture["sha256"], "inventory_input_changed")
    helper_identity = file_identity(copied_helper.lstat())
    need(value["source"]["sha256"] == HELPER_SHA256 and value["source"]["before"] == value["source"]["after"] == helper_identity
         and all(value.get("checks", {}).get(k) is True for k in
                 ("rootIdentityVerified", "sourceUnchanged", "inputUnchanged", "inputDescriptorClosed")), "inventory_source_or_closure")
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
             and row["encodedBytes"] > 0 and position + row["encodedBytes"] <= fixture["bytes"]
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
             and offset % alignment == 0 and tensor["absoluteDataOffset"] == data_start + offset <= fixture["bytes"]
             and tensor["descriptor"] == {"offset": position + begin, "encodedBytes": cursor - begin,
                 "encodedSha256": sha(raw[begin:cursor])}, "ordered_tensor_descriptor_binding")
    need(cursor == len(raw) and len({t["name"] for t in observed}) == tensors, "descriptor_inventory_complete")
    need(value["tensorData"]["offset"] == data_start and value["tensorData"]["payloadBytesReadForDecoding"] == 0
         and value["tensorData"]["payloadLengthsInferred"] is False, "observer_payload_limit")
    return {"headerBound": True, "allMetadataHashesBound": True, "orderedDescriptorsBound": True,
            "metadataEntries": metadata, "tensors": tensors, "numericTensorTypes": dict(collections.Counter(str(t["ggmlType"]) for t in observed))}

def tensor_bytes(tensor):
    dimensions, kind = tensor["dimensions"], tensor["ggmlType"]
    elements = math.prod(dimensions)
    need(kind in (0, 1, 2), "unknown_encoding_no_byte_guess")
    if kind == 2:
        need(dimensions[0] % 32 == 0, "q4_row_block_shape")
        return elements // 32 * 18
    return elements * (4 if kind == 0 else 2)

def validate_inspection(fixture, inventory, value):
    scalar = lambda key: inventory["allowlistedScalars"].get(key, {}).get("value")
    need(value.get("schema_version") == 1 and value.get("kind") == "quantopsy.container-inspection"
         and value.get("file") == {"name": fixture["name"], "size": fixture["bytes"], "sha256": fixture["sha256"]}
         and value.get("architecture") == scalar("general.architecture") and value.get("model_name") == scalar("general.name")
         and value.get("quantization_version") == scalar("general.quantization_version")
         and value.get("alignment") == inventory["tensorData"]["alignment"]
         and value.get("metadata_entries") == inventory["header"]["metadataCount"]
         and value.get("evidence") == "container_structure_and_bytes" and value.get("behavioral_validation") == "not_run", "application_inspection_binding")
    names = {0: "F32", 1: "F16", 2: "Q4_0"}
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

def observe(fixture):
    owner()
    observation = {"fixture": fixture, "observerComplete": False,
                   "applicationAdmission": {"accepted": False, "status": "unrun"}}
    report["observations"].append(observation)
    file = root / "models" / fixture["name"]
    verify_inputs("before_" + fixture["id"])
    out, err, row = run("inventory_" + fixture["id"], [str(Path(sys.executable).resolve()), "-I", "-B", "-S",
                        str(copied_helper), "--fixture", fixture["helperFixture"], "--input", str(file)], 100, allowed=(0, 2), cwd=root / "output")
    marker = b"QUANTOPSY_PUBLIC_GGUF_INVENTORY_RESULT="
    need(out.startswith(marker) and out.endswith(b"\n") and out.count(marker) == 1, "inventory_original_marker")
    value = strict_json(out[len(marker):-1])
    observation["inventory"] = value
    need(row["exitCode"] == 0 and err == b"", "independent_inventory_refused")
    observation["inventoryBindings"] = validate_inventory(fixture, value)
    observation["observerComplete"] = True
    verify_inputs("after_inventory_" + fixture["id"])
    out, err, row = run("inspect_" + fixture["id"], [str(Path(sys.executable).resolve()), "-B", "-S", "-m", "quantopsy", "inspect", str(file)], 90, allowed=(0, 2))
    admission = observation["applicationAdmission"]
    admission["exitCode"] = row["exitCode"]
    if row["exitCode"] == 0:
        need(err == b"", "unexpected_application_stderr")
        inspection = strict_json(out)
        admission["inspection"] = inspection
        validate_inspection(fixture, value, inspection)
        admission.update({"accepted": True, "status": "admitted"})
    else:
        need(out == b"", "refusal_stdout_not_empty")
        failure = strict_json(err)
        admission["refusal"] = failure
        need(type(failure) is dict and set(failure) == {"error"} and type(failure["error"]) is dict
             and set(failure["error"]) == {"code", "message"}, "application_refusal_shape")
        code, message = failure["error"]["code"], failure["error"]["message"]
        need(isinstance(code, str) and code.startswith("gguf_") and len(code) <= 100
             and all(c.islower() or c.isdigit() or c == "_" for c in code) and isinstance(message, str)
             and 0 < len(message.encode()) <= 1024 and not any(ord(c) < 32 or ord(c) == 127 for c in message), "abnormal_application_refusal")
        admission.update({"accepted": False, "status": "refused", "originalNonzeroExitRetained": True})
    verify_inputs("after_inspect_" + fixture["id"])


def evaluate_pair(observations):
    import re
    result = {"accepted": False, "status": "not_evaluated", "selectedTensor": SELECTED_TENSOR,
              "acceptanceScope": "selected_pair_structural_and_opaque_byte_prerequisites_only",
              "requirements": {}, "blockers": [], "unquantizedBindings": [],
              "selectedNumericalValidation": "not_evaluated", "provenanceLineage": "not_verified",
              "repairPlanCreated": False, "behavioralValidation": "not_run"}
    def requirement(name, value, code):
        result["requirements"][name] = value
        if value is not True:
            result["blockers"].append(code)
    rows = []
    for fixture in FIXTURES[1:]:
        matches = [row for row in observations if row["fixture"]["id"] == fixture["id"]]
        if len(matches) != 1 or matches[0].get("observerComplete") is not True:
            result["blockers"].append("pair_inventory_not_observed")
            return result
        rows.append(matches[0])
    paths = [root / "models" / row["fixture"]["name"] for row in rows]
    before = [fingerprint(file, row["fixture"]) for file, row in zip(paths, rows)]
    inventories = [row["inventory"] for row in rows]
    maps = [{tensor["name"]: tensor for tensor in inv["tensors"]} for inv in inventories]
    metadata = [{entry["key"]: entry for entry in inv["metadata"]} for inv in inventories]
    ref, quant = maps
    result["status"] = "evaluated"
    requirement("completeIndependentInventoriesObserved", True, "pair_inventory_not_observed")
    supported = all(tensor["ggmlType"] in (0, 1, 2) for inv in inventories for tensor in inv["tensors"])
    requirement("currentGlobalTypesSupported", supported, "pair_global_encoding_unsupported")
    requirement("referenceOnlyF32F16", all(t["ggmlType"] in (0, 1) for t in ref.values()), "pair_reference_encoding_unsupported")
    admitted = all(row["applicationAdmission"]["accepted"] is True for row in rows)
    requirement("bothApplicationInspectionsAdmitted", admitted, "pair_application_admission_refused")
    architecture = all(inv["allowlistedScalars"].get("general.architecture") == {"kind": 8, "value": "llama"} for inv in inventories)
    requirement("bothArchitecturesLlama", architecture, "pair_architecture_not_llama")
    excluded = {"general.file_type", "general.quantization_version", "general.quantized_by", "general.quantized_at"}
    metadata_bindings = [{key: (entry["encodedBytes"], entry["encodedSha256"]) for key, entry in entries.items() if key not in excluded}
                         for entries in metadata]
    requirement("completeEncodedMetadataEqualOutsideQuantizationKeys", metadata_bindings[0] == metadata_bindings[1], "pair_encoded_metadata_mismatch")
    same = set(ref) == set(quant) and all(ref[name]["dimensions"] == quant[name]["dimensions"] for name in set(ref) & set(quant))
    requirement("completeTensorNamesAndDimensionsEqual", same, "pair_tensor_inventory_mismatch")
    unsupported = re.compile(r"expert|(?:^|[._])exps?(?:[._]|$)|moe|lora|adapter|mmproj|vision|audio|attn_qkv|ffn_gate_up", re.ASCII | re.IGNORECASE)
    dense = True
    for inv in inventories:
        for entry in inv["metadata"]:
            if "expert" in entry["key"] or "moe" in entry["key"]:
                dense = dense and (entry["key"].endswith((".expert_count", ".expert_used_count"))
                         and entry.get("valueRetained") is True and type(entry.get("scalarValue")) is int and entry["scalarValue"] == 0)
        dense = dense and all(len(t["dimensions"]) <= 2 and not unsupported.search(t["name"]) for t in inv["tensors"])
    requirement("denseLlamaMetadataAndLayoutsEstablished", dense, "pair_dense_layout_unresolved_or_unsupported")
    a, b = ref.get(SELECTED_TENSOR), quant.get(SELECTED_TENSOR)
    selected = a is not None and b is not None and a["ggmlType"] == 1 and b["ggmlType"] == 2 and a["dimensions"] == b["dimensions"] and len(a["dimensions"]) == 2 and a["dimensions"][0] % 32 == 0
    requirement("explicitSelectedF16Q4RankTwo", selected, "selected_f16_q4_rank_two_not_established")
    elements = math.prod(a["dimensions"]) if selected else None
    requirement("selectedWorkWithin64M", elements is not None and elements <= 64000000, "selected_work_not_established_or_over_bound")
    result["selectedObservation"] = {"reference": None if a is None else {"ggmlType": a["ggmlType"], "dimensions": a["dimensions"]},
                                   "quantized": None if b is None else {"ggmlType": b["ggmlType"], "dimensions": b["dimensions"]},
                                   "elements": elements}
    if admitted and supported and same:
        inspection_maps = [{t["name"]: t for t in row["applicationAdmission"]["inspection"]["tensors"]} for row in rows]
        equal = True
        for name in sorted(quant):
            q, r = quant[name], ref[name]
            if q["ggmlType"] not in (0, 1):
                continue
            if q["ggmlType"] != r["ggmlType"]:
                equal = False
                continue
            reference_bytes, quantized_bytes = inspection_maps[0][name]["bytes"], inspection_maps[1][name]["bytes"]
            if reference_bytes != quantized_bytes:
                equal = False
                continue
            reference_sha = hash_span(paths[0], r["absoluteDataOffset"], reference_bytes)
            quantized_sha = hash_span(paths[1], q["absoluteDataOffset"], quantized_bytes)
            result["unquantizedBindings"].append({"name": name, "ggmlType": q["ggmlType"], "bytes": quantized_bytes,
                                                  "referenceSha256": reference_sha, "quantizedSha256": quantized_sha})
            equal = equal and reference_sha == quantized_sha
        requirement("allUnquantizedTypesAndPayloadHashesEqual", equal, "unquantized_type_or_payload_mismatch")
    else:
        requirement("allUnquantizedTypesAndPayloadHashesEqual", None, "unquantized_byte_bindings_not_evaluated")
    after = [fingerprint(file, row["fixture"]) for file, row in zip(paths, rows)]
    need(before == after, "pair_inputs_changed")
    requirement("inputsCurrentBeforeAndAfterOpaqueReads", True, "pair_inputs_changed")
    result["accepted"] = all(value is True for value in result["requirements"].values())
    return result


def preflight():
    global root, identity, marker_identity, nonce, env, baseline, work_end, copied_helper
    need(sys.version_info >= (3, 11) and sys.platform == "linux" and sys.flags.isolated and sys.flags.no_site
         and sys.dont_write_bytecode, "isolated_preinstalled_python_required")
    need(source_commit is not None and len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit), "source_commit_required")
    need(os.environ.get("CI") == "true" and os.environ.get("GITHUB_ACTIONS") == "true"
         and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY and os.environ.get("GITHUB_REF") == BRANCH
         and os.environ.get("GITHUB_EVENT_NAME") == "create" and os.environ.get("QUANTOPSY_ARTIFACT_PUBLIC_REPOSITORY") == "true", "approved_public_ci_required")
    event = strict_json(read(os.environ["GITHUB_EVENT_PATH"], 262144))
    need(event.get("ref_type") == "branch" and event.get("ref") == BRANCH.removeprefix("refs/heads/")
         and event.get("repository", {}).get("private") is False
         and event.get("repository", {}).get("full_name") == REPOSITORY, "approved_create_event_required")
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
    root = Path(tempfile.mkdtemp(prefix="qg-", dir=parent))
    root.chmod(0o700)
    info = root.lstat()
    identity = (info.st_dev, info.st_ino)
    nonce = secrets.token_hex(16)
    fd = os.open(root / ".owner", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(nonce.encode())
    marker = (root / ".owner").lstat()
    marker_identity = (marker.st_dev, marker.st_ino)
    for name in ("models", "tmp", "output", "config", "cache", "data", "state"):
        (root / name).mkdir(mode=0o700)
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
           "QUANTOPSY_ARTIFACT_INVENTORY_SHA256": HELPER_SHA256}
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
    helper = read(repo / HELPER, 128 * 1024)
    need(sha(helper) == HELPER_SHA256, "independent_helper_source_changed")
    copied_helper = root / "output" / "public_gguf_inventory.py"
    fd = os.open(copied_helper, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(helper)
    report["generated"] = {"helperSourcePath": HELPER, "sourceSha256": HELPER_SHA256,
                           "copiedSha256": sha(read(copied_helper, 128 * 1024)), "copiedByteExact": read(copied_helper, 128 * 1024) == helper,
                           "identity": file_identity(copied_helper.lstat())}
    need(report["generated"]["copiedByteExact"], "independent_helper_copy_changed")

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
        verify_inputs("before_removal")
        allowed = {".owner": {"sha256": sha(nonce.encode()), "bytes": 32}}
        if copied_helper is not None and copied_helper.exists():
            allowed[str(copied_helper.relative_to(root))] = {"sha256": HELPER_SHA256, "bytes": copied_helper.stat().st_size}
        for fixture in FIXTURES:
            if fixture["id"] in tracked_files:
                allowed["models/" + fixture["name"]] = {"sha256": fixture["sha256"], "bytes": fixture["bytes"]}
        for download_row in report["downloads"]:
            partial = download_row.get("partialFile")
            if partial is not None:
                allowed[partial["relativePath"]] = {"bytes": partial["writtenBytes"], "sha256": partial["writtenSha256"],
                                                   "createdDevice": partial["createdDevice"], "createdInode": partial["createdInode"]}
        actual = set()
        expected_directories = {"models", "tmp", "output", "config", "cache", "data", "state"}
        checks = []
        for parent, directories, names in os.walk(root, followlinks=False):
            for name in directories:
                directory = Path(parent) / name
                need(str(directory.relative_to(root)) in expected_directories and not directory.is_symlink()
                     and directory.stat().st_uid == os.getuid() and stat.S_IMODE(directory.stat().st_mode) == 0o700, "unexpected_directory_preserved")
            for name in names:
                file = Path(parent) / name
                relative = str(file.relative_to(root))
                need(relative in allowed, "unexpected_file_preserved")
                expected = allowed[relative]
                if "createdInode" in expected:
                    current_identity = file_identity(file.lstat())
                    need(current_identity["device"] == expected["createdDevice"] and current_identity["inode"] == expected["createdInode"], "partial_path_replaced")
                current = fingerprint(file, expected)
                if relative.startswith("models/"):
                    original = next(value for value in tracked_files.values() if value["relativePath"] == relative)
                    need(current == original, "model_changed_before_removal")
                if relative == "output/public_gguf_inventory.py":
                    need(file_identity(file.lstat()) == report["generated"]["identity"], "helper_changed_before_removal")
                checks.append(current)
                actual.add(relative)
        need(actual == set(allowed), "owned_file_set_changed")
        storage_sample("before_removal")
        owner()
        result["ownedFileChecks"] = checks
        shutil.rmtree(root)
        need(not root.exists(), "scratch_still_present")
        report["scratchCleanup"] = "passed"
        guard("after_cleanup", cleanup=True, post_scratch=True)
        rows, uncertain = owned()
        result.update({"remaining": rows, "uncertain": uncertain,
                       "allOriginalGroupsAbsent": all(group_gone(pid) for pid in groups),
                       "allChildrenReaped": all(p.poll() is not None for p in processes)})
        need(not rows and not uncertain and result["allOriginalGroupsAbsent"] and result["allChildrenReaped"], "post_cleanup_process_remaining")
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
        for fixture in FIXTURES:
            download(fixture)
        need(len(report["downloads"]) == 3 and sum(x["receivedBytes"] for x in report["downloads"]) == DOWNLOAD_TOTAL
             and all(x["complete"] and x["bodyEOFObserved"] and x["responseClosed"] and x["fileDescriptorClosed"] and not x["resourceErrors"] for x in report["downloads"]), "downloads_incomplete")
        verify_inputs("after_downloads")
        guard("after_downloads")
        for fixture in FIXTURES:
            observe(fixture)
        need(len(report["observations"]) == 3 and all(x["observerComplete"] for x in report["observations"]), "three_inventories_required")
        report["pairRestorationAdmission"] = evaluate_pair(report["observations"])
        verify_inputs("after_pair_observation")
        need(read(copied_helper, 128 * 1024) and sha(read(copied_helper, 128 * 1024)) == HELPER_SHA256
             and file_identity(copied_helper.lstat()) == report["generated"]["identity"], "helper_copy_changed")
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
        report["commandOutcomes"] = [{"label": row["label"], "exitCode": row["exitCode"],
                                     "classification": "container_refusal_observed" if row["label"].startswith("inspect_") and row["exitCode"] == 2
                                     else "zero_exit" if row["exitCode"] == 0 else "unexpected_exit"}
                                    for row in report["commands"]]
        normal_outcomes = all(x["classification"] in ("zero_exit", "container_refusal_observed") for x in report["commandOutcomes"])
        report["processCleanup"] = {"allRowsNormallyClosed": lifecycle, "allCommandExitsZero": all(row["exitCode"] == 0 for row in report["commands"]),
                                    "originalCommands": len(report["commands"]), "signalsSent": any(row["termSent"] or row["killSent"] for row in report["commands"])}
        report["diagnosticComplete"] = bool(completed and not report["errors"] and not interrupted and lifecycle and normal_outcomes
                                             and report.get("cleanup", {}).get("passed") is True and report.get("scratchCleanup") == "passed"
                                             and len(report["guards"]) == 5 and all(g["passed"] for g in report["guards"])
                                             and work_end is not None and time.monotonic() <= work_end + 90)
        report["packetStatus"] = "observed" if report["diagnosticComplete"] else "failed"
        report["completedAtUtc"] = utc()
        output = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        if len(output) > REPORT_LIMIT:
            report["packetStatus"], report["diagnosticComplete"] = "failed", False
            report["errors"].append("report_bound_parsed_duplicates_omitted")
            for observation in report["observations"]:
                if "inventory" in observation:
                    observation["inventory"] = {"omittedParsedCopy": True, "originalRetainedInCommand": "inventory_" + observation["fixture"]["id"]}
                if "inspection" in observation["applicationAdmission"]:
                    observation["applicationAdmission"]["inspection"] = {"omittedParsedCopy": True, "originalRetainedInCommand": "inspect_" + observation["fixture"]["id"]}
            output = json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        need(len(output) <= REPORT_LIMIT, "report_bound")
        signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
        signal.setitimer(signal.ITIMER_REAL, 2)
        sys.stdout.buffer.write(b"QUANTOPSY_PUBLIC_GGUF_OBSERVATION_RESULT=" + output + b"\n")
        sys.stdout.buffer.flush()
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0 if report["diagnosticComplete"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
