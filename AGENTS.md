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
  No tests have been approved for this new repository yet.
- Read the supplied storage policy and run Storage Guard before large checkpoint
  copies, installs or builds. Do not bypass a failure or clean unrelated work.
- Keep contracts, provenance and evidence in docs/. State source publication,
  format checks, inference/behavior, runner and distribution limits separately.
