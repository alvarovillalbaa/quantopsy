# Frozen comparison and first inference packet

8 November 2026. Standing owner approval covers this free source/verification
scope. The [brief](../spec.md) requires behavior evidence; the previous
[v2 result](../results/2026/11-07/gguf-compat-v2-attempt-1.md) proves bytes only.
The [comparison command](../behavior.md) imports records without inference;
matching hashes alone cannot authenticate execution or prior task freezing.

Use `ggml-org/models-moved` revision
`4cd12190a40543c61bad59a1b8d982dfc48de020`, directory `tinyllamas`:

| Role | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Reference | stories15M.gguf | 98,357,920 | `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a` |
| Quantized | stories15M-q4_0.gguf | 19,077,344 | `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04` |

Predeclare all 42 matrices: layers 0–5, attention q/k/v/output and FFN
gate/up/down. Restore exact F32 reference payloads under explicit v2. Expected
selected elements: 5,971,968; growth: 20,528,640 bytes. Require fresh plan/repair
and independent all-57-payload verification. Preserve embedding/output Q8 and
all unselected bytes. No adaptive selection or repeated candidates.

Build official llama.cpp b5046 commit
`74d4f5b041ad837153b0e90fc864b8290e01d8d5` and the authored CI-only scorer.
CPU only, two threads/build jobs, context/batch 128, explicit BOS, no EOS,
no sampling, seed 0 unused, FP32 KV; no flash attention, CURL or OpenMP.
Refuse context extrapolation, truncation, nonfinite scores and joint tokenization
that changes the prompt prefix. Hash a canonical engine-source manifest binding
scorer, aggregation adapter, CMake and upstream revision. Retain actual source
archive/binary hashes and build identity separately. An observed archive hash
is not a previously trusted publisher signature.

The official commit API binds source tree `9ae6f9dd0984b780c942ccf510e9cd5fcee9e724`.
Reconstruct that Git tree from archive contents before building. Its one pinned
gitlink is `ggml/src/ggml-kompute/kompute` at
`4565194ed7c32d1d2efa32ceab4d3c6cae006306`; require no submodule payload and
never fetch it. KOMPUTE stays disabled. The complete recursive API tree reports
1,416 entries, no symlinks and `truncated: false`.

Freeze eight independently authored short story continuations and their order,
configuration and thresholds before the first forward pass. Teacher forcing
scores target tokens from the true preceding text. Preserve target/predicted
token IDs and per-token NLL; all roles use identical engine/config/tokenizer.
These tasks do not measure general ability or instruction following.

| Metric | Better | Minimum quantization loss | Minimum recovered fraction | Maximum regression vs reference | Maximum regression vs quantized |
| --- | --- | --- | --- | --- | --- |
| mean_target_nll | Lower | 0.005 | 0.5 | 0.02 | 0.01 |
| target_token_accuracy | Higher | 0.02 | 0.5 | 0.03 | 0.02 |

Apply thresholds per task/metric pair. Require complete runs, at least one
qualifying loss, recovery of every qualifying loss and no excessive regression.
No qualifying loss never means recovery. Preserve negative results; never tune
tasks, thresholds or candidate after outputs in this packet.

Run new authored comparison checks and 16 relevant existing CLI cases only;
do not repeat the closed 92-case packet or unrelated diagnostics. Freshly hash
downloads, verify pair/candidate, build and run all roles. Freeze actual model,
binary and record identities before inference, recheck afterwards and retain
chronological receipts plus bounded original stdout/stderr.

One free public Ubuntu job, at most 20 minutes. Source archive at most 64 MiB,
extracted source 512 MiB/build 2 GiB, two build jobs, no package installation or
model API. Keep HOME intact. Use owned scratch, exact source/ref guards,
create-only files, attributed children, EOF/reaping checks and verified teardown
on success/failure. Fixed-path hash-bound reuse of the old operational controller
is allowed; its old main/closed scope must never run. Retire temporary CI.

Local Storage Guard reports 10.1 GiB free versus 34.0 GiB required on 8 November.
No local weights, installs, app/model execution or builds. Static AST/whitespace
review and evidence handling remain allowed. No paid plan/account, private or
production data, provider/model API, production config or database is involved.

## Changed control after the first failure

The [first invocation](../results/2026/11-08/behavioral-inference-attempt-1.md)
passed all 51 cases, actual candidate payload checks and the engine build, then
failed build-tree file admission before freezing/inference. Cleanup is unverified.
Preserve the exact originals; no model result exists to tune against.

The second control changes only owned build-file admission and finite refusal
diagnostics, retaining strict ordinary input restrictions and later identity
checks. It binds the first receipt and every unchanged application/test input
instead of repeating the closed 51 cases. Fresh model/candidate/build/inference
and verified cleanup remain mandatory. No task, metric, threshold, selection,
engine scorer, provider, cost or product runtime scope changes.

The [second result](../results/2026/11-08/behavioral-inference-attempt-2.md)
completed the unchanged experiment and verified cleanup. Recovery failed the
frozen criteria (11 qualifying losses, 10 recovered, four regressions). This
protocol is closed; preserve both attempts rather than changing thresholds or
tasks to reverse the outcome.
