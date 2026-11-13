# Explicit input-embedding v3 — fixture qualification

The new capability is qualified for the authored file/record scope at source
`27ea65a5d0d7e65cae18f6a8380e6e8af78f09e3`, tree
`979e729818b588299ca6b438d61db38f4fba36b4`. All 54 source-input hashes match
the second invocation's complete vector. Five product modules add explicit v3
admission, embedding shape/selection and manifest binding; default v1 and explicit
v2 remain supported. No runtime dependency or inference engine was added.

The [contract](../../../proposals/input-embedding-v3.md) scopes 25 new v3 cases,
28 v2 integration regressions, 16 CLI cases and five existing manifest cases.
All **74 distinct identities passed across two invocations** on existing native
Python 3.14.7. There was no single 74-pass invocation:

| Invocation | Actual child outcome | Later observation |
| --- | --- | --- |
| 1 | Exit 1; 74 exact names, 72 passes and two failures; unittest reports 0.236 s | The new cases expected a later repair-layer error. Both inputs were already refused as `gguf_nonfinite_weight`. The outer observer then failed at `spawnSync ps EPERM`; its failed status is retained. |
| 2 | Exit 0; only the two corrected cases selected, both passed; unittest reports 0.003 s | Source/index guards passed. A separate native process/open-file observation and owned-root removal completed. |

Only the two expected error literals changed between invocations. No product,
shared fixture, helper or other test logic changed. Fifty-three of 54 source
inputs are byte-identical; the previous [v3 test source](profile-v3-test-source-attempt-1.py.txt)
is retained. Exact literal replacement and every other file hash were checked
before carrying 72 passed identities. Those cases were not replayed. The 28 v2
cases use `v4` as their unknown future-profile sentinel; deliberately supporting
v3 does not relax earlier positive behavior or rewrite historical receipts.

The [qualification JSON](profile-v3-checks.json) owns the full accepted name set,
stream identities and carry proof. Original [observation 1](profile-v3-checks-attempt-1.json)
remains failed, with [data-only reconciliation](profile-v3-checks-reconciliation-1.json);
[observation 2](profile-v3-checks-attempt-2.json) records the actual selected
execution and [separate cleanup](profile-v3-checks-cleanup-2.json). Exit and close
events were captured for both children. Both before/after 54-file vectors and
Git index hashes matched. Native `ps` found no remaining members of the exact
owned groups; native `lsof` reported no open files under either private fixture
root. Both nonce/device/inode-checked roots were removed and read back absent.
These are scoped snapshots, not all-host or past-descendant absence proof.

All original streams were retained before source and cleanup checks. The first
stdout is empty; its 16,986-byte stderr has SHA-256
`fffea03c0cbdb92c5d9a0d42f2d8936c83ef5df2c050f84be5656d5bf974dc2e`.
It is preserved locally in the owning delivery notes. Publication excludes local
absolute paths: the [16,854-byte representation](profile-v3-checks-attempt-1.stderr.redacted.txt)
replaces exactly four repository-path occurrences with `[repository]`, with its
own hash in the reconciliation. The original receipt's stderr filename/count/hash
describe the original at collection, not this public representation. No original
bytes were reconstructed or a redacted digest mislabeled as the original.
The second [stdout](profile-v3-checks-attempt-2.stdout.txt) is empty and its
[stderr](profile-v3-checks-attempt-2.stderr.txt) retains both actual passing lines.

The tests exercise real tiny independently encoded GGUF files, exact whole-file
candidate bytes, unselected opaque Q8 preservation, explicit cross-version
refusals, width/token/vocabulary constraints, finite selected values, budgets,
tampering, create-only paths, real CLI readback and data-only manifest seams.
They use no network/model fixture, account, provider API, paid service,
dependency install, build, browser, database or CI. Storage Guard refused heavy
local work at 3.6 GiB free / 34 GiB required; no exception was used for weights,
an engine, a build or an environment.

The [behavioral study v2](../../../studies/embedding-v2.md) and its [frozen JSON](../../../studies/embedding-v2.json)
remain **UNRUN**, SHA-256
`8c35578fb0f8a2451a876f0a86cacf98941f7887a289df0bfd19c5ae416c5bc3`.
Eight new prompt/target pairs are disjoint from the original eight strings, and
both metric/threshold records exactly match the old study. Actual tokenizer
lengths, four fresh role runs, two candidates, engine/controller identity,
compression tradeoff and new recovery verdict remain future verification gates.
The exact bounded CI controller has not yet been authored or activated.
The previous negative result is unchanged; no new behavioral recovery or full
product/portfolio completion follows from these fixture passes.
