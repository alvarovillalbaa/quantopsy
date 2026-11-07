# Opt-in v2: opaque Q8_0 and F32 restoration

IMPLEMENTED; the [first v2 receipt](../results/2026/11-07/gguf-compat-v2-attempt-1.md)
verifies the scoped byte-level path. Product acceptance remains false.
The [actual observation](../results/2026/11-07/public-gguf-observation-attempt-2.md)
admitted the 15M reference as F32, refused its Q4-labelled partner because
`output.weight` is Q8_0, and established matching metadata/name/dimension maps.
It did **not** evaluate the 13 unquantized payload bindings or create a plan.

## Explicit compatibility choice

Add `--profile v2` to `inspect`, `plan` and `repair`; omission retains v1 behavior
and schemas exactly. V2 inspection, plans and repair manifests use
`schema_version: 2`, existing kind identifiers and strict fields. Repair requires
the explicit matching profile/version; reject cross-version plans rather than
silently upgrading. `diagnose` and `probe` remain v1. Preserve v1 examples,
validators and results; document the new choice separately.

## Container and restoration rules

V2 admits F32/F16/Q4_0 plus **opaque, unselected Q8_0** throughout the inventory.
The [pinned block definition](https://github.com/ggml-org/llama.cpp/blob/eb4e244be77a906a183cd4a00fd51d9ad3200955/ggml/src/ggml-common.h#L251-L257)
specifies 32 values, a two-byte scale and 32 signed bytes: 34 bytes per block.
Require complete blocks along dimension zero; derive spans from this contract,
then retain alignment, overlap, file-bound and quantization-version checks.
Never infer encoding sizes from gaps. `iter_values` must dispatch exact types
and explicitly refuse Q8 numerical decoding, not fall through to Q4 logic.

V2 plans restore only explicitly selected rank-two dense-Llama Q4_0 matrices
from matching **F32** reference bytes. V1's selected F16 contract stays intact.
Keep complete name/dimension equality; encoded metadata equality outside
`general.file_type`, `general.quantization_version`, `general.quantized_by` and
`general.quantized_at`; an entirely F32/F16 reference; and identical type and
full payload hash for every unquantized F32/F16 tensor in the quantized input.
Establish all 13 bindings before accepting this pair.

Preserve explicit selection, dense/expert/fused/adapter fences, provenance,
finite selected-value checks, full input hashes, work/growth/storage budgets,
source ownership/change detection. Copy every unselected payload, including
Q8, unchanged. Reparse the candidate under v2 and verify every prescribed tensor
hash. Preserve create-only output, tracked-inode cleanup, misleading-file-type
omission and manifest-last completion; a bundle is not an atomic transaction.

## Owners and verification gate

- [contracts.py](../../quantopsy/contracts.py): explicit type/profile constants;
  [gguf.py](../../quantopsy/gguf.py): versioned admission, Q8 span and decode refusal.
- [repair.py](../../quantopsy/repair.py): separate strict v2 plan validation,
  selected F32 binding and writer revalidation; [cli.py](../../quantopsy/cli.py):
  opt-in/version dispatch; [format contract](../format.md): both profiles.
- New tiny authored fixtures cover valid Q8 spans and opaque preservation;
  incomplete rows, truncation/overlap and decode refusal; F32/Q4 v2 restoration;
  metadata/dimension/unquantized-hash/version/budget/input-change refusals; and
  create-only/manifest-last failures. Run relevant existing v1 reader, repair
  and CLI regressions for these seams. The old 80-case receipt is historical;
  the full historical 80-case suite was not repeated.

After source review, one owned free-CI packet uses the diagnostic's exact pinned
15M downloads: v2 inspect, explicit `blk.0.attn_q.weight` plan, repair and
candidate inspection. Require the fresh 13 bindings, unchanged complete input
hashes, selected F32 hash and every unselected output hash, final manifest,
bounded originals and normal process/EOF/artifact cleanup. Retain no weights in
Git. No dependencies, automatic selection, inference, behavior recovery or
cryptographic-lineage claim is added. The first invocation passed at `e18dee05`:
92 fixture cases, 13 fresh unquantized bindings and all 57 prescribed output
payloads, with normal closure and verified teardown. The disposed candidate was
19,362,432 bytes. Preserve the unchanged earlier v1 refusal and original outputs;
inference and recovered behavior remain separate, untested gates.
