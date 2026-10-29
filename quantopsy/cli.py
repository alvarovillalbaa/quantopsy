"""Local commands; no model engine, credentials or network service."""

import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import sys

from .contracts import DEFAULT_LIMITS, TYPE_NAMES
from .diagnostics import diagnose, probe
from .errors import QuantopsyError, fail
from .gguf import GGUFFile
from .io import canonical_json_bytes, read_json, write_json_create
from .repair import create_plan, validate_plan, write_candidate


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        fail("invalid_arguments", "Arguments are missing or invalid. Use --help for supported commands and options.")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="quantopsy",
        description="Inspect paired GGUF weights and write an explicit restoration candidate.",
        epilog="Numerical diagnostics and restored tensors do not establish recovered model behavior.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Read a supported GGUF container and its complete hash.")
    inspect.add_argument("model")
    inspect.add_argument("--output", help="Create a new JSON file instead of printing the report.")

    def paired(name: str, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("reference", help="Regular unquantized F32/F16 GGUF file.")
        command.add_argument("quantized", help="Regular compatible F32/F16/Q4_0 GGUF file.")
        return command

    compare = paired("diagnose", "Compare explicitly selected tensor values.")
    compare.add_argument("--tensor", action="append", required=True)
    compare.add_argument("--max-elements", type=int, default=DEFAULT_LIMITS.max_selected_elements)
    compare.add_argument("--output")
    linear = paired("probe", "Compare linear responses to supplied calibration vectors.")
    linear.add_argument("--probes", required=True, help="Versioned local linear-probe JSON.")
    linear.add_argument("--max-elements", type=int, default=DEFAULT_LIMITS.max_selected_elements)
    linear.add_argument("--output")
    plan = paired("plan", "Bind an explicit selected-tensor restoration intent.")
    plan.add_argument("--tensor", action="append", required=True)
    plan.add_argument("--provenance", required=True, help="Declared model/revision/license/source JSON.")
    plan.add_argument("--max-added-bytes", type=int, required=True)
    plan.add_argument("--max-elements", type=int, default=DEFAULT_LIMITS.max_selected_elements)
    plan.add_argument("--output", required=True, help="Create a new bound repair-plan JSON file.")
    repair = paired("repair", "Verify a bound plan and create a new GGUF candidate bundle.")
    repair.add_argument("--plan", required=True)
    repair.add_argument("--output-dir", required=True, help="Must not already exist, including empty directories.")
    repair.add_argument("--reserve-bytes", type=int, default=1024**3,
                        help="Minimum free bytes after estimated output; default 1 GiB.")
    return parser


def _inspect(model: GGUFFile) -> dict[str, object]:
    def scalar(key: str) -> object:
        entry = model.metadata.get(key)
        return entry.value if entry and type(entry.value) in (str, int, float, bool) else None

    return {
        "schema_version": 1, "kind": "quantopsy.container-inspection",
        "file": {"name": model.path.name, "size": model.file_size, "sha256": model.sha256()},
        "architecture": scalar("general.architecture"), "model_name": scalar("general.name"),
        "quantization_version": scalar("general.quantization_version"),
        "alignment": model.alignment, "metadata_entries": len(model.metadata),
        "tensor_types": dict(Counter(TYPE_NAMES[tensor.kind] for tensor in model.tensors)),
        "tensors": [{"name": tensor.name, "dimensions": list(tensor.dimensions),
                     "type": TYPE_NAMES[tensor.kind], "bytes": tensor.nbytes,
                     "elements": tensor.elements} for tensor in model.tensors],
        "evidence": "container_structure_and_bytes", "behavioral_validation": "not_run",
    }


def _emit(value: object, output: str | None) -> None:
    if output is not None:
        write_json_create(output, value)
        print(canonical_json_bytes({"saved_file": Path(output).name,
                                    "status": "created"}).decode("utf-8"), end="")
    else:
        print(canonical_json_bytes(value).decode("utf-8"), end="")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        with ExitStack() as resources:
            if args.command == "inspect":
                model = resources.enter_context(GGUFFile(args.model))
                _emit(_inspect(model), args.output)
                return 0
            reference = resources.enter_context(GGUFFile(args.reference))
            quantized = resources.enter_context(GGUFFile(args.quantized))
            if args.command == "diagnose":
                result = diagnose(reference, quantized, args.tensor, max_elements=args.max_elements)
            elif args.command == "probe":
                result = probe(reference, quantized, read_json(args.probes), max_elements=args.max_elements)
            elif args.command == "plan":
                result = create_plan(reference, quantized, args.tensor, read_json(args.provenance),
                                     args.max_added_bytes, max_elements=args.max_elements)
            else:
                plan = validate_plan(read_json(args.plan))
                result = write_candidate(reference, quantized, plan, args.output_dir,
                                         reserve_bytes=args.reserve_bytes)
            _emit(result, getattr(args, "output", None))
        return 0
    except QuantopsyError as error:
        print(canonical_json_bytes({"error": {"code": error.code,
                                              "message": str(error)}}).decode("utf-8"),
              file=sys.stderr, end="")
        return 2
    except OSError:
        print('{"error":{"code":"io_error","message":"An input or output could not be accessed. Check paths and available space."}}',
              file=sys.stderr)
        return 2
