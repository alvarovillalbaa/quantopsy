# Authored fixture checks — 80 passed

2 November 2026. **80/80 passed** in one local invocation at immutable
[Quantopsy source `b6f3552`](https://github.com/alvarovillalbaa/quantopsy/tree/b6f355290df7cdb5ed673dbf5865cf4f99dfe0e9).
The persistent owner approval and root's exact assignment covered the existing
[four-module proposal](../../../proposals/fixture-checks.md). No product/test
repair was needed. [JSON receipt](fixture-checks.json) retains all 80 actual
identities/statuses, the 25 input fingerprints and runtime/output evidence.

| Requested module | Passed |
| --- | ---: |
| `tests.test_gguf` | 26/26 |
| `tests.test_diagnostics` | 16/16 |
| `tests.test_repair` | 22/22 |
| `tests.test_cli` | 16/16 |

Existing **Python 3.14.7**, selected by its resolved executable, ran:

```sh
python3 -B -S -m unittest tests.test_gguf tests.test_diagnostics tests.test_repair tests.test_cli -v
```

`-B` prevents bytecode writes; `-S` disables site startup. These runner-only
restrictions do not change the requested modules or assertions. A child
preflight reported the same interpreter/version and flags under the sanitized
environment. Its binary SHA-256 was
`87d4df53fd91304be5bac391fb204643c36b7df2023c04a0953bcbc7d4fdf634`
and remained unchanged. Published identities use a portable executable basename.

Observed UTC: **2026-11-02T08:38:53.024929+00:00** to **2026-11-02T08:38:53.565302+00:00**.
Actual exit **0**, no signal/stop/interruption; the process was reaped. Verbose
unittest output reports **Ran 80 tests / OK**, with 80 unique identities matching
the frozen inventory and zero failures, errors, skips, expected failures or
unexpected successes. The 120-second wall cap, TERM/KILL three-second grace,
three-second post-kill drain bound and 256 KiB cap per output stream did not
trigger. Only explicit PATH/TMPDIR/TZ and Python startup/encoding controls were
passed; no HOME, account/provider credentials or Python path override was inherited.

All **25 sealed source files** matched their original SHA-256/size and canonical
Git mode/blob before execution and remained unchanged afterward. The original
[inventory](../10-29/source-inventory.json) is unchanged. Independently authored
GGUF fixtures are bounded to 64 KiB by the fixture encoder. No real checkpoint,
trained weights or inference engine was used. The fixture directory was empty
after the suite; its device/inode and outer owner marker/identity were verified
before the owned temporary tree was removed. Bytecode creation was disabled;
no existing pycache, unrelated file or environment was deleted.

The subsequent publication edition updates only `AGENTS.md`, `README.md` and
`docs/proposals/fixture-checks.md` to replace stale approval/status wording and
link these results. Those three of the 25 historical documentation inputs
therefore differ after execution. Product code, tests, fixtures, specification
and the original seal remain unchanged; the status-document edits were not rerun.

Stdout was empty. Raw verbose stderr was **14,705 bytes**, SHA-256
`9cbdac6aedda7b4a34e6c07252b87630bee256fd25b957108cb33f92d7597ba9`.
This proves the focused GGUF/diagnostic/selected-byte-repair/CLI fixture contract
for the observed runtime. It does not establish real model lineage, licensed
checkpoint compatibility, inference loading, recovered behavior, regression
thresholds or portability. No install, build, CI, hosted/account/database,
network/provider/model call, deployment or paid action occurred. These receipts
are local owning-repository evidence; root owns publication. The full Quantopsy
and portfolio objective remains open.
