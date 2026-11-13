# Explicit input-embedding restoration, compatibility v3

13 November 2026. The [negative first inference study](../results/2026/11-08/behavioral-inference-attempt-2.md)
restored 42 internal matrices but kept `token_embd.weight` quantized. Its retained
independent inventory identifies the input embedding as Q4_0, 288 × 32,000,
5,184,000 bytes, and the output head as opaque Q8_0, 9,792,000 bytes. This does
not establish which tensor caused the observed regressions. Input-embedding
restoration is a new hypothesis, not a repair to the old experiment's verdict.

The original tool's [full objective](../spec.md) still requires credible behavior
evidence. This opt-in product capability enables a separately frozen follow-up;
successful fixture or candidate-byte checks cannot replace that inference.

## Contract

`inspect`, `plan` and `repair` accept explicit `--profile v3`. V1 and v2 retain
their existing target selection and record versions. A v3 operation requires
both readers and its plan to use v3; a plan cannot silently select its profile.
Reports/plans/manifests use version 3. The data-only behavioral comparator can
bind those manifests without claiming to execute or authenticate inference.

V3 has v2's F32/F16/Q4_0 container support and opaque, unselected Q8_0 spans.
It restores exact F32 payloads into selected Q4_0 dense Llama matrices, adding
only the explicit target `token_embd.weight`. It does not add output-head or Q8
restoration, infer tied weights, select targets automatically, or decode Q8.
`diagnose` and `probe` keep their existing profile.

The input embedding must be rank two with dimensions exactly
`(llama.embedding_length, len(tokenizer.ggml.tokens))`. Width must be positive
uint32 metadata; tokens must be a nonempty string array. An optional declared
`llama.vocab_size` must agree with that count and type. Paired metadata/dimensions
still match exactly; all input hashes, finite selected values, budgets,
create-only output, unselected bytes and manifest-last checks remain.

Primary format guidance: [pinned GGUF names/metadata](https://github.com/ggml-org/ggml/blob/facdc44a67da17df4433adcc468b424f48e9ded2/docs/gguf.md)
and [pinned Llama tensor construction](https://github.com/ggml-org/llama.cpp/blob/eb4e244be77a906a183cd4a00fd51d9ad3200955/src/models/llama.cpp).
The latter creates token embeddings with embedding-width and vocabulary axes.
The v3 requirement for a tokenizer list and consistency with optional explicit
vocabulary size is a conservative local admission policy.

## Focused verification

The owner's standing “approve everything / keep going / do all the work”
authority covers these necessary source changes and the concrete 74-case
fixture scope: 25 new `ProfileV3IntegrationTier2Tests`, 28 v2 integration cases,
16 CLI cases and five existing behavior-manifest cases named below. The changed
reader/profile/repair/CLI/manifest seams justify those regressions. No closed
80/92/51-case packet or model experiment is replayed unchanged.

Only v2's invalid-future-profile sentinel changes from `v3` to `v4`, because v3
is now deliberately supported. No older positive assertion or original receipt
is weakened. The new independent GGUF fixtures stay below 64 KiB and cover
exact candidate bytes, opaque output preservation, explicit versions, shape,
finite values, tampering, budgets, readback, output ownership and actual CLI
file seams. Invented behavior records remain explicitly unauthenticated.

The five existing `BehaviorComparisonTier2Tests` selected are:

- `test_good_measurements_without_repair_manifest_cannot_pass_bound_recovery_records`
- `test_both_supported_repair_manifest_versions_bind_supplied_records`
- `test_manifest_full_reference_quantized_and_candidate_identities_must_match`
- `test_manifest_cannot_bind_an_arbitrary_or_stale_canonical_plan_hash`
- `test_manifest_requires_strict_plan_arithmetic_verification_and_unclaimed_behavior`

Use existing Python 3.14.7 for this lightweight local fixture profile, bytecode
disabled and fixtures confined to one owned temporary directory. Bound the child
to 60 seconds and each original stream to 256 KiB; retain exit, every case
status, source/index fingerprints and cleanup. This is a new runtime/source
profile, not another execution at the historical Linux/Python 3.12.3 source.

Storage Guard measured 3.6 GiB free / 34 GiB required. No local checkpoint,
model/engine execution, install, build or large copy is permitted. The tiny
fixture scope needs no environment or storage exception. No network, paid
resource, account, provider/model API, production setting or database is used.
At preparation the separate [frozen behavioral follow-up](../studies/embedding-v2.md)
was unexecuted. Its [later actual result](../results/2026/11-13/embedding-study-v2-attempt-1.md)
completed four role runs; both candidates failed the unchanged recovery criteria.

## Fixture outcome

The [actual qualification](../results/2026/11-13/profile-v3-checks.md) accepts
74 distinct identities across two invocations at exact source `27ea65a5`.
The first run had 72 passes and two wrong new error-code expectations; its
observer then hit `ps EPERM`. Only those two cases reran after correcting their
expectations to the decoder's earlier nonfinite refusal. Full product/shared
source and all other test logic stayed exact. Original failures and separate
native cleanup/readbacks remain explicit; no single 74-pass run or inference
acceptance is claimed. This fixture scope is closed.
