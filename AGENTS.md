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
