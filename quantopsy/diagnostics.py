"""Paired numerical evidence, distinct from end-to-end model behavior."""

from __future__ import annotations

import hashlib
import itertools
import json
import math

from .contracts import DEFAULT_LIMITS, TYPE_NAMES, TensorInfo
from .errors import fail
from .gguf import GGUFFile
from .io import canonical_json_bytes
from .repair import pair_inputs

MAX_SELECTION = 256
MAX_PROBE_VECTORS = 16
MAX_PROBE_COLUMNS = 16384


class _Sum:
    def __init__(self) -> None:
        self.value = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        adjusted = value - self.correction
        total = self.value + adjusted
        if not math.isfinite(total):
            fail("numerical_overflow", "The supplied numerical work exceeds finite arithmetic.")
        self.correction = (total - self.value) - adjusted
        self.value = total


class _Norm:
    """Accumulate a compensated sum of squares relative to the largest value."""

    def __init__(self) -> None:
        self.scale = 0.0
        self.squares = _Sum()

    def add(self, value: float) -> None:
        magnitude = abs(value)
        if not math.isfinite(magnitude):
            fail("numerical_overflow", "The supplied numerical work exceeds finite arithmetic.")
        if magnitude == 0:
            return
        if magnitude > self.scale:
            previous = self.squares
            ratio = self.scale / magnitude
            self.squares = _Sum()
            # Rebase both the normalized total and its retained low-order error.
            self.squares.add(previous.value * ratio * ratio)
            self.squares.add(-previous.correction * ratio * ratio)
            self.squares.add(1.0)
            self.scale = magnitude
        else:
            ratio = magnitude / self.scale
            self.squares.add(ratio * ratio)

    def rms(self, count: int) -> float:
        # Dividing the normalized sum first avoids overflowing a norm when the
        # requested RMS itself is representable. Scale stays nonzero for tiny
        # inputs even when their raw squares would underflow to zero.
        value = self.scale * math.sqrt(self.squares.value / count)
        if not math.isfinite(value) or (value == 0 and self.scale != 0):
            fail("numerical_overflow", "The nonzero requested RMS is outside the supported floating-point range.")
        return value

    def relative_to(self, reference: _Norm) -> float | None:
        if reference.scale == 0:
            return None
        if self.scale == 0:
            return 0.0
        factor = math.sqrt(self.squares.value / reference.squares.value)
        numerator, numerator_exp = math.frexp(self.scale)
        denominator, denominator_exp = math.frexp(reference.scale)
        multiplier, multiplier_exp = math.frexp(factor)
        try:
            value = math.ldexp((numerator / denominator) * multiplier,
                               numerator_exp - denominator_exp + multiplier_exp)
        except OverflowError:
            fail("numerical_overflow", "The requested relative L2 exceeds finite arithmetic.")
        if not math.isfinite(value) or value == 0:
            fail("numerical_overflow", "The nonzero requested relative L2 is outside the supported floating-point range.")
        return value


class _Metrics:
    def __init__(self) -> None:
        self.count = 0
        self.changed = 0
        self.error_norm = _Norm()
        self.reference_norm = _Norm()
        self.maximum_error = 0.0

    def add(self, reference: float, candidate: float) -> None:
        if not math.isfinite(reference) or not math.isfinite(candidate):
            fail("nonfinite_values", "Numerical diagnostics require finite values.")
        error = reference - candidate
        self.error_norm.add(error)
        self.reference_norm.add(reference)
        self.maximum_error = max(self.maximum_error, abs(error))
        self.count += 1
        self.changed += reference != candidate

    def result(self) -> dict[str, object]:
        if self.count == 0:
            fail("empty_values", "Numerical diagnostics require at least one value.")
        return {
            "elements": self.count,
            "changed_values": self.changed,
            "rmse": self.error_norm.rms(self.count),
            "reference_rms": self.reference_norm.rms(self.count),
            "max_absolute_error": self.maximum_error,
            "relative_l2": self.error_norm.relative_to(self.reference_norm),
            "relative_l2_unavailable": "zero_reference_energy" if self.reference_norm.scale == 0 else None,
        }


def _budget(value: int, reference: GGUFFile, quantized: GGUFFile) -> int:
    maximum = min(reference.limits.max_selected_elements,
                  quantized.limits.max_selected_elements)
    if type(value) is not int or value < 1 or value > maximum:
        fail("work_limit", "The numerical work budget is outside the supported limit.")
    return value


def _selection(reference: GGUFFile, quantized: GGUFFile,
               names: list[str]) -> list[tuple[TensorInfo, TensorInfo]]:
    if type(names) is not list or not 1 <= len(names) <= MAX_SELECTION:
        fail("invalid_selection", "Select between one and 256 tensor names explicitly.")
    if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
        fail("invalid_selection", "Tensor selection must contain distinct nonempty names.")
    return [(reference.tensor(name), quantized.tensor(name)) for name in names]


def diagnose(reference: GGUFFile, quantized: GGUFFile, names: list[str], *,
             max_elements: int = DEFAULT_LIMITS.max_selected_elements) -> dict[str, object]:
    budget = _budget(max_elements, reference, quantized)
    selected = _selection(reference, quantized, names)
    elements = sum(left.elements for left, _right in selected)
    if elements > budget:
        fail("work_limit", "The selected tensors exceed the numerical work budget.")
    compatibility = pair_inputs(reference, quantized)
    reports: list[dict[str, object]] = []
    missing = object()
    for left, right in selected:
        metrics = _Metrics()
        for a, b in itertools.zip_longest(reference.iter_values(left), quantized.iter_values(right),
                                           fillvalue=missing):
            if a is missing or b is missing:
                fail("tensor_changed", "A selected tensor has an unexpected value count.")
            metrics.add(a, b)
        if metrics.count != left.elements:
            fail("tensor_changed", "A selected tensor has an unexpected value count.")
        reports.append({"name": left.name, "dimensions": list(left.dimensions),
                        "reference_type": TYPE_NAMES[left.kind],
                        "quantized_type": TYPE_NAMES[right.kind], **metrics.result()})
    reference.assert_unchanged()
    quantized.assert_unchanged()
    return {"schema_version": 1, "kind": "quantopsy.weight-diagnosis",
            "inputs": compatibility, "selected_elements": elements,
            "tensors": reports, "evidence": "parameter_error_only",
            "behavioral_validation": "not_run"}


def _probe_document(value: object) -> tuple[str, list[list[float]], dict[str, str]]:
    expected = {"schema_version", "kind", "tensor", "vectors", "provenance"}
    if type(value) is not dict or set(value) != expected or type(value["schema_version"]) is not int \
            or value["schema_version"] != 1 or value["kind"] != "quantopsy.linear-probes":
        fail("invalid_probes", "Use the versioned linear-probe document shape.")
    tensor, vectors, provenance = value["tensor"], value["vectors"], value["provenance"]
    if type(tensor) is not str or not tensor or len(tensor) > 256:
        fail("invalid_probes", "A linear-probe tensor name is required.")
    keys = {"model_id", "revision", "dataset", "capture"}
    if type(provenance) is not dict or set(provenance) != keys \
            or any(type(item) is not str or not item.strip() or len(item) > 1024
                   for item in provenance.values()):
        fail("invalid_probes", "Supply bounded model, revision, dataset and capture provenance.")
    if type(vectors) is not list or not 1 <= len(vectors) <= MAX_PROBE_VECTORS:
        fail("invalid_probes", "Supply between one and sixteen calibration vectors.")
    checked: list[list[float]] = []
    for vector in vectors:
        if type(vector) is not list or not 1 <= len(vector) <= MAX_PROBE_COLUMNS:
            fail("invalid_probes", "Calibration vectors exceed the supported width.")
        row: list[float] = []
        for item in vector:
            if type(item) not in (int, float):
                fail("invalid_probes", "Calibration vectors require finite numeric values.")
            try:
                number = float(item)
            except (ValueError, OverflowError):
                fail("invalid_probes", "Calibration vectors require finite numeric values.")
            if not math.isfinite(number):
                fail("invalid_probes", "Calibration vectors require finite numeric values.")
            row.append(number)
        checked.append(row)
    return tensor, checked, dict(provenance)


def probe(reference: GGUFFile, quantized: GGUFFile, document: object, *,
          max_elements: int = DEFAULT_LIMITS.max_selected_elements) -> dict[str, object]:
    budget = _budget(max_elements, reference, quantized)
    probe_bytes = canonical_json_bytes(document)
    snapshot = json.loads(probe_bytes)
    name, vectors, provenance = _probe_document(snapshot)
    selected = _selection(reference, quantized, [name])
    left, right = selected[0]
    if len(left.dimensions) != 2 or left.dimensions != right.dimensions:
        fail("invalid_probes", "Linear probes require a matching rank-two tensor pair.")
    columns, rows = left.dimensions
    if any(len(vector) != columns for vector in vectors):
        fail("invalid_probes", "Calibration vector width must match the tensor's first dimension.")
    work = left.elements * len(vectors)
    if work > budget:
        fail("work_limit", "The calibration vectors exceed the paired linear-work budget.")
    compatibility = pair_inputs(reference, quantized)
    reference_values, quantized_values = reference.iter_values(left), quantized.iter_values(right)
    metrics = [_Metrics() for _ in vectors]
    try:
        for _ in range(rows):
            a = tuple(itertools.islice(reference_values, columns))
            b = tuple(itertools.islice(quantized_values, columns))
            if len(a) != columns or len(b) != columns:
                fail("tensor_changed", "A selected tensor has an unexpected value count.")
            for vector, result in zip(vectors, metrics, strict=True):
                actual = math.fsum(x * y for x, y in zip(a, vector, strict=True))
                candidate = math.fsum(x * y for x, y in zip(b, vector, strict=True))
                result.add(actual, candidate)
    except (ValueError, OverflowError):
        fail("numerical_overflow", "The supplied linear probes exceed finite arithmetic.")
    if next(reference_values, None) is not None or next(quantized_values, None) is not None:
        fail("tensor_changed", "A selected tensor has an unexpected value count.")
    reference.assert_unchanged()
    quantized.assert_unchanged()
    return {"schema_version": 1, "kind": "quantopsy.linear-probe-result",
            "inputs": compatibility, "tensor": name, "dimensions": list(left.dimensions),
            "provenance": provenance,
            "canonical_probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
            "paired_work_elements": work,
            "vectors": [{"index": index, **result.result()} for index, result in enumerate(metrics)],
            "evidence": "supplied_linear_probes_only", "behavioral_validation": "not_run"}
