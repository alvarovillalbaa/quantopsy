# Proposed 80 local authored-fixture checks

29 October 2026. **Prepared, not approved or executed.** Source publication is
authorized separately by the portfolio goal. The supplied workspace and local
`AGENTS.md` require explicit approval for domain/CLI execution; AST parsing and
read-only source review do not grant that permission.

## Exact action

Use the existing Python 3.14.7 executable from the same shell, with no install
or dependency change. From this repository, run only:

```sh
python3 -m unittest tests.test_gguf tests.test_diagnostics tests.test_repair tests.test_cli -v
```

No glob, broad discovery, other project or real checkpoint. Each GGUF fixture
is independently encoded with `struct`, bounded to 64 KiB, in a disposable local
temporary directory. The fixture models have no inference behavior. CLI cases
call the actual entry function under captured streams and authored local files;
they do not spawn an inference engine or install a package. Controlled failures
use mocks to verify cleanup and ownership boundaries.

| Explicit file | Literal cases | Purpose |
| --- | ---: | --- |
| `tests/test_gguf.py` | 26 | Container admission, metadata/bounds, alignment, F16/F32/Q4_0 byte order, source/descriptor ownership and malformed inputs |
| `tests/test_diagnostics.py` | 16 | Paired weight metrics, supplied linear responses, selection/work/provenance/finite-result boundaries |
| `tests/test_repair.py` | 22 | Compatibility, strict bound plan, actual selected-byte restoration, unchanged payloads, metadata/offsets, tampering, overwrite refusal, capacity and owned cleanup |
| `tests/test_cli.py` | 16 | Five command paths, finite errors, bounded JSON, create-only records and resource cleanup |
| **Total** | **80** | **All new, authored and currently unrun** |

The [source inventory](../results/2026/10-29/source-inventory.json) freezes all
80 literal method names and exact file hashes. No dynamic test discovery was
performed to prepare that inventory.

## Recording and repair boundary

One supervised local process is capped at 120 seconds. Record the approval,
source commit, Python version/binary hash, exact arguments, frozen input hashes,
start/end, exit/signal, raw stdout/stderr hashes and actual pass/fail/error/skip
counts under this repository's dated results directory. Expected discovery is
exactly 80; missing/extra cases cannot count as success. Retain failed attempts.

If that approved run exposes a defect in this exact source slice, make the
smallest bounded repair and rerun the same 80 cases, recording updated hashes
and every attempt. Added cases, broader suites, other runtimes or real-model
execution require a separate scope. Remove only disposable fixtures created by
these cases; do not touch user data or existing outputs.

No install, dependency, build/package, server, browser, database, migration, CI,
account, hosted/provider configuration, network/model call, downloaded model,
real weights/activation data, inference, promotion, deployment, payment or paid
enrollment is included. The source CLI needs no Vercel/Supabase service.

Passing these checks would establish fixture behavior for one observed local
runtime. It would not verify Python/platform portability, real model lineage,
licensed checkpoint compatibility, inference loading, recovered behavior or
regression thresholds. Those full-project gates remain open in the
[product contract](../spec.md).
