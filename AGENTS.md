# Quantopsy

- Owning intent is the Notion Quantopsy brief: investigate and repair behavior
  lost through model quantization. Numerical weight/probe diagnostics and a new
  mixed-precision candidate are useful source work, not proof of recovered model
  behavior. Keep frozen reference/quantized/candidate inference acceptance open.
- Source creation and publication to personal GitHub are authorized by the
  portfolio goal. Preserve unrelated work and use apply_patch for manual edits.
- Runtime uses Python 3.11+ standard library only. No model/service dependency,
  network request, pickle, remote code, plugins or subprocess inference runner.
  New dependencies, Auth/security policy, CI and model execution need approval.
- First supported container is single-file little-endian GGUF v3, F32/F16/Q4_0.
  Reject unsupported types, malformed metadata, shards, bad dimensions, unaligned
  or overlapping spans and files outside documented budgets. Decode from pinned
  primary format contracts; never infer byte sizes for unknown encodings.
- Repair is explicit selected Q4_0 dense Llama matrices restored from matching
  F16 reference payloads. Do not mutate either input or automatically select a
  tensor for repair. Preserve unselected payloads, recompute aligned offsets,
  omit misleading general.file_type and retain quantization_version when needed.
- Pair compatibility and declared revision/license provenance do not establish
  cryptographic lineage. Bind both complete input hashes and planned per-tensor
  hashes; detect changing sources and verify the rebuilt file before success.
- Outputs are create-only. Do not overwrite existing paths or follow output
  symlinks. A manifest written last marks a complete candidate bundle; do not
  describe multiple files as one atomic transaction. Cleanup only owned files.
- Never commit weights, real prompts/probes, local-only paths, credentials or
  user artifacts. Tests use independently authored tiny GGUF byte fixtures.
- AST parsing and diff/whitespace/import checks are allowed. Domain/backend
  checks and model/CLI fixture execution require their explicit scoped approval.
  The owner's continued approval covered the 80-case authored fixture scope;
  its [2 November result](docs/results/2026/11-02/fixture-checks.md) records the
  passing run and its proof limits.
- Read the supplied storage policy and run Storage Guard before large checkpoint
  copies, installs or builds. Do not bypass a failure or clean unrelated work.
- Keep contracts, provenance and evidence in docs/. State source publication,
  format checks, inference/behavior, runner and distribution limits separately.

## Approved public GGUF observations — 7 November 2026

The owner's standing "I approve everything / Keep going / Do all the work"
covers the [separate public-artifact packet](docs/proposals/public-gguf-observation.md):
one temporary free public Ubuntu job, at most 15 minutes, on
`qa/quantopsy-artifacts-20261107-1`. Read only three explicitly pinned public
GGUF fixtures (118,620,640 bytes total), independently observe their bounded
headers/inventories, and run the existing `inspect` CLI once per file.
Do not rerun the closed 80-case fixture suite, change application code, install
dependencies, execute model code/inference, create probes/repairs or use accounts.

Diagnostic completion is separate from each actual container-admission result
and from F16/Q4_0 restoration-pair admission. A refusal stays false with original
exit/output; no pre-authored positive pair or behavior-recovery claim is allowed.
Keep both full input hashes, actual descriptor evidence and all source/output
limits. Preserve HOME; use owned runner-only weight/temp paths, verify unchanged
inputs, reaping/EOF/groups and scoped teardown. Never commit weight/tokenizer
payloads. Disable/retire the workflow after its terminal run; another attempt
requires a concrete changed repair. Local Storage Guard still forbids heavy
operations below reserve; no local download, install, build or runtime exception.

## Approved documented manual control — candidate 2

The create-event source remained [unrun](docs/results/2026/11-07/public-gguf-observation-attempt-1.md);
no runtime failure, model-file result or cause is inferred. Standing approval
covers the concrete dispatch-only control change. Default branch contains a
constant-false inert stub at `.github/workflows/public-gguf-observation-2.yml`;
it performs no checkout or observation. The reviewed QA activation supplies
the executable workflow only for `qa/quantopsy-artifacts-20261107-2`, exact
workflow_dispatch event/repository/public ref, and fixed reviewed SOURCE_COMMIT.
No dispatch inputs may choose code, input files or runner arguments.

Keep the inert stub byte-exact in checked source, preserve the first source/note,
and retain every domain/helper/input/hash/resource limit from candidate 1.
The first actual invocation still needs all original outputs and owned teardown.
Disable the owned workflow and retire the active QA body after a terminal run;
remove the temporary default stub only through the final reviewed publication.
No authentication, Actions permission, dependency, app-code, file, inference
or paid-service scope is broadened.

The [actual public-file receipt](docs/results/2026/11-07/public-gguf-observation-attempt-2.md) records diagnostic completion
at `a604b0df2c176630ddcc8874477dea7e797ce01f`, with two admitted F32
containers and one retained Q8_0 refusal. Restoration-pair and product admission
remain false; no unquantized payload equality, candidate or behavior is proved.
The owned workflow is disabled. First control remains UNRUN; keep both scopes
closed and their source/input/output identities intact before new work.
The temporary default stub is retired in the later publication; its presence
and hash above describe the checked preparation only.

## Approved opt-in v2 compatibility slice — 7 November 2026

Standing owner approval covers the separate [v2 contract](docs/proposals/opaque-q8-f32-v2.md):
explicit `--profile v2` on inspect/plan/repair, opaque unselected Q8_0 spans,
and selected dense-Llama Q4_0 restoration from matching F32 references.
Preserve default v1 admission, selected F16 behavior, schemas and historical
results. Both readers and each operation must explicitly agree on profile;
strict plan versions cannot select or silently upgrade the operation profile.
Q8 numerical decoding, selected Q8 restoration and v2 diagnose/probe stay out
of scope. All provenance, pairing, hash, budget and output-ownership fences stay.

The changed reader/repair/CLI seams justify one new authored Tier 2 integration
suite plus the existing 64 reader/repair/CLI cases in one owned free public
Ubuntu CI packet, capped at 15 minutes. The unchanged 16 diagnostic cases and
closed public-observation scopes are not repeated. Then inspect the exact two
pinned 15M files under v2, bind every unquantized payload freshly, and explicitly
plan/repair/inspect `blk.0.attn_q.weight`. Preserve original outputs and failures;
do not infer pair admission from matching metadata. A hash-verified candidate
is byte-level evidence, never inference or recovered behavior.

No local runtime/download/install/build exception is granted. Keep HOME and
all unrelated work intact. No dependencies, model code, probes, automatic
selection, account, paid service or provider call. Own every temporary input,
candidate and process; require bounded original output, unchanged source/input
identities and verified teardown. Disable and retire the owned CI control after
its terminal run. Source review and static checks precede any execution.

The [first v2 receipt](docs/results/2026/11-07/gguf-compat-v2-attempt-1.md)
records the completed invocation at `e18dee0559949f0bbedba01da9c3006b228c44cd`:
92 fixture passes, fresh pair bindings and a hash-verified candidate, with
normal process closure and owned artifact teardown. The workflow is disabled.
Preserve this source and its immutable originals; do not repeat the closed
scope without a concrete changed need. Product acceptance remains false and
model inference/behavior recovery remain untested.

## Approved frozen behavioral slice — 8 November 2026

Standing owner approval covers the [behavioral contract](docs/proposals/behavioral-inference-v1.md):
a data-only comparison command, focused authored checks and 16 relevant CLI
regressions, then one temporary free public Ubuntu job capped at 20 minutes.
The application stays Python-standard-library-only; it never downloads or
executes an engine. Only the reviewed CI adapter may build pinned official
llama.cpp and the authored CPU scorer with existing compiler/CMake.

Predeclare 42 matrices, eight authored story tasks, teacher-forced metrics,
thresholds and settings before output. Bind scorer/adapter/build sources,
actual binary, tokenizer, all models and frozen records. Preserve all tasks,
failures, regressions and token measurements; no outcome-based tuning.
Execution completion and recovery criteria are separate outcomes. Imported JSON
proves neither inference nor prior freezing; this tiny suite cannot establish
broad semantic or general model recovery. Keep closed packets unchanged.

Own and verify runner downloads/build/model outputs/processes and teardown;
disable and retire temporary CI after the terminal invocation. No local storage
exception, paid API, account, provider, production data or product dependency
change is included.

The [first behavioral invocation](docs/results/2026/11-08/behavioral-inference-attempt-1.md)
passed all 51 cases and candidate/source/build checks, then refused build-tree
admission before inference. Preserve its originals and unverified teardown.
Standing approval covers a changed second control: admit ordinary 0644/0755
tool-generated build leaves only beneath the private owned build tree, retaining
exact identity/mode/hash on later checks; ordinary model/source inputs keep
their existing restrictions. Record finite refusal details. Reuse the closed
51-case receipt only after binding unchanged application/test inputs; no repeated
test or model-quality pass is implied. Fresh build/candidate/inference and owned
teardown remain required. The predeclared experiment is unchanged.

The [second behavioral invocation](docs/results/2026/11-08/behavioral-inference-attempt-2.md)
completed 24 actual task/role measurements and verified owned teardown at
`40b0e846`. Its frozen decision is NEGATIVE: 11 qualifying losses, 10 recovered,
four regressions. The 51 exact-input checks are retained first-run evidence,
not rerun. Preserve both originals and the negative protocol; do not retune it
or describe it as semantic/general recovery. This scope is closed. Temporary
controls are disabled/retired; final publication removes default registration.
Future strategy changes need a separate versioned predeclared study rather
than overwriting these results. Full product acceptance remains open.

## Explicit embedding profile and frozen follow-up — 13 November 2026

The owner's standing all-work approval covers the separate [v3 contract](docs/proposals/input-embedding-v3.md)
and its 74 lightweight authored fixture checks using the existing Python, no
install/build/weights/network. V3 alone can explicitly restore a Q4_0
`token_embd.weight` from matching F32, with width/token-count consistency.
V1/v2 targets, defaults, record versions and closed receipts remain unchanged;
v2's invalid future-profile test uses v4 rather than the newly valid v3.
Q8/output-head restoration, implicit selection and inference in the product
remain prohibited. Full source, per-case status, original streams and owned
fixture cleanup are required. Storage Guard still blocks heavy local work.

The [new behavioral study v2](docs/studies/embedding-v2.md) predeclares disjoint
authored tasks and compares fresh reference/quantized/legacy42/embedding43
measurements under the old thresholds. It is UNRUN until its exact bounded CI
controller and source/engine/candidate freeze are ready. Preserve the first
negative study and every failure. Fixture or byte success never supplies the
missing new inference result; no positive/general recovery claim is authorized
by this preparation. No paid resource, account or model/provider API is involved.

The [fixture qualification](docs/results/2026/11-13/profile-v3-checks.md) accepts
74 distinct cases at source `27ea65a5`: 72 first-run passes plus two corrected
decoder-refusal expectations, not one all-green 74-case execution. The first
observer's `ps EPERM`, failed child, originals and explicit public path redactions
are preserved. Product/shared fixtures stayed exact; only two error literals
changed before the two-case completion. Native snapshots found both owned groups
absent; both roots were removed. Never rerun the 72 for a green outer packet. No model
ran, and the new behavioral study/controller remains unrun/unactivated.

## Exact embedding-study-v2 execution — 13 November 2026

Standing all-work approval covers the new [execution packet](docs/proposals/embedding-study-v2-execution.md):
one free public Ubuntu job capped20min, exact parent b60d04db, four added source/
proposal/template files and this append only. Frozen JSON8c35578f, all product/
test/old-study/scorer/build bytes remain exact. Carry 74 fixture identities with
source/two-literal proofs; execute no fixture suite. Freshly bind both public
inputs, rebuild legacy42/v3embedding43 candidates, reconstruct/build the pinned
official CPU engine, freeze both triplets and perform four actual role runs on
the eight new tasks. Preserve all originals and both threshold verdicts.

Controller/helper source is separate; old main/protocol never runs or changes.
No model/API/payment/account or local heavy exception. Enforce source/index,
process/EOF/groups, immutable inputs, exact storage/budgets and nonce/inode
teardown. A negative study may complete operationally; no global/semantic
recovery claim follows. Retire only this one exact workflow/QA ref after terminal
proof. Full product/portfolio remains unfinished.
