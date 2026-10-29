"""Bounded data-only JSON and create-only writes."""

import json
import math
import os
from pathlib import Path
import stat

from .errors import QuantopsyError, fail

JSON_BYTES = 2 * 1024**2


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            fail("invalid_json", "JSON objects must not contain duplicate keys.")
        result[key] = value
    return result


def _constant(_value: str) -> object:
    fail("invalid_json", "JSON numbers must be finite.")


def read_json(path: str | Path, max_bytes: int = JSON_BYTES) -> object:
    if type(max_bytes) is not int or max_bytes < 1:
        fail("invalid_limit", "The JSON byte limit must be a positive integer.")
    descriptor = None
    try:
        if os.name != "posix":
            fail("json_platform_unsupported", "JSON file inputs currently support POSIX macOS and Linux.")
        leaf = os.lstat(path)
        if not stat.S_ISREG(leaf.st_mode):
            fail("invalid_json_file", "JSON inputs must be regular files, not symlinks or streams.")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (leaf.st_dev, leaf.st_ino):
            fail("invalid_json_file", "The JSON input changed while it was opened.")
        if before.st_size > max_bytes:
            fail("json_too_large", "The JSON input exceeds its byte limit.")
        source = os.fdopen(descriptor, "rb")
        descriptor = None
        with source:
            data = source.read(max_bytes + 1)
            after = os.fstat(source.fileno())
        current = os.lstat(path)
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                value.st_mtime_ns, value.st_ctime_ns)
        if not stat.S_ISREG(current.st_mode) or fields(before) != fields(after) or fields(before) != fields(current):
            fail("invalid_json_file", "The JSON input changed while it was read.")
        if len(data) > max_bytes:
            fail("json_too_large", "The JSON input exceeds its byte limit.")
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=_constant)
        _validate(value)
        return value
    except QuantopsyError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        fail("invalid_json", "The input must be bounded, valid UTF-8 JSON.")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate(value: object, depth: int = 0, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [0]
    budget[0] += 1
    if depth > 16 or budget[0] > 100_000:
        fail("invalid_json", "The JSON structure exceeds its depth or entry limit.")
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                fail("invalid_json", "JSON object keys must be text.")
            try:
                key.encode("utf-8")
            except UnicodeError:
                fail("invalid_json", "JSON text must contain valid Unicode.")
            _validate(item, depth + 1, budget)
    elif type(value) is list:
        for item in value:
            _validate(item, depth + 1, budget)
    elif type(value) not in (str, int, float, bool, type(None)):
        fail("invalid_json", "Only ordinary JSON values are supported.")
    elif type(value) is float and not math.isfinite(value):
        fail("invalid_json", "JSON numbers must be finite.")
    elif type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError:
            fail("invalid_json", "JSON text must contain valid Unicode.")


def canonical_json_bytes(value: object, max_bytes: int = JSON_BYTES) -> bytes:
    if type(max_bytes) is not int or max_bytes < 1:
        fail("invalid_limit", "The JSON byte limit must be a positive integer.")
    _validate(value)
    try:
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, UnicodeError, RecursionError):
        fail("invalid_json", "The output contains an unsupported JSON value.")
    if len(data) > max_bytes:
        fail("json_too_large", "The JSON output exceeds its byte limit.")
    return data


def write_json_create(path: str | Path, value: object, max_bytes: int = JSON_BYTES) -> None:
    data = canonical_json_bytes(value, max_bytes)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    identity = None
    try:
        identity = os.fstat(descriptor)
        target = os.fdopen(descriptor, "wb")
        descriptor = None
        with target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        if identity is None:
            fail("cleanup_incomplete", "A failed JSON write could not establish output ownership; inspect the output before retrying.")
        try:
            current = os.lstat(path)
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            fail("cleanup_incomplete", "A failed JSON write left an owned partial file; inspect the output before retrying.")
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
