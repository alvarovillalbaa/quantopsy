# Quantopsy

Investigate quantization loss and create a local weight-restoration candidate.

The source inspects GGUF containers, compares selected weights and supplied
linear activation probes, and restores explicitly selected Q4_0 tensors from
an F16 reference into a NEW GGUF bundle. It uses Python 3.11+ and the standard
library on POSIX macOS/Linux, with no model download, hosted service or inference
engine. No Python/platform runtime matrix has been verified yet.

**Behavioral recovery is not established.** Parameter error and supplied linear
probes do not measure end-to-end task quality. A created candidate still needs
frozen reference/quantized/candidate inference and regression checks. The
[product contract](docs/spec.md) preserves that full objective.

## Use from source

No pip installation is required. From the repository root:

```sh
python3 -m quantopsy inspect quantized-q4_0.gguf
python3 -m quantopsy diagnose reference-f16.gguf quantized-q4_0.gguf \
  --tensor blk.0.attn_q.weight --output diagnosis.json
python3 -m quantopsy probe reference-f16.gguf quantized-q4_0.gguf \
  --probes local-probes.json --output linear-probes.json
python3 -m quantopsy plan reference-f16.gguf quantized-q4_0.gguf \
  --tensor blk.0.attn_q.weight --provenance local-provenance.json \
  --max-added-bytes 67108864 --output repair-plan.json
python3 -m quantopsy repair reference-f16.gguf quantized-q4_0.gguf \
  --plan repair-plan.json --output-dir candidate-bundle
```

These are usage examples, not observed model runs. Supply your own compatible,
licensed local files. The [provenance example](examples/provenance.example.json)
contains required fields, not verified model identity or a weight license.
The [probe example](examples/probes.example.json) demonstrates a tiny 32-value
input shape only; it must be replaced with relevant captured activation vectors
of the selected tensor's first dimension.

Plans bind both full file hashes, selected tensor hashes, caller-declared
revision/license provenance and an explicit added-weight/work budget. Repair
rechecks those bindings and verifies the rebuilt candidate. Existing output
paths, including empty directories and symlinks, are refused. The manifest is
written last; use a bundle only when its completion manifest matches the model.
An interrupted multi-file write is not an atomic transaction.

## First supported profile

- Single regular GGUF v3 little-endian files, with only F32/F16/Q4_0 tensors.
- Dense `llama` selected attention/FFN matrices: Q4_0 candidate and F16 reference.
- Eight GiB per input; bounded metadata and 64 million selected work elements.
- No shards, streams, symlink inputs, nested metadata arrays, tokenizer-only
  files, unknown encodings, expert/adapter/multimodal restoration or model code.

Other architectures can be inspected within the container profile; restoration
is deliberately narrower. Check [format and provenance](docs/format.md) before
using a plan. Model weights are not included and the software license grants no
rights to third-party weights.

Run your storage preflight before large checkpoints or outputs. Repair checks
estimated output capacity and leaves 1 GiB free by default; `--reserve-bytes`
does not replace a stricter local reserve policy. Neither input is rewritten.

## Verification status

Source authoring/static checks and publication are recorded in the
[delivery receipt](docs/results/2026/10-29/initial-source.md). All 80 authored
GGUF, diagnostics, repair and CLI fixture checks passed on Python 3.14.7 at
`b6f355290df7cdb5ed673dbf5865cf4f99dfe0e9`; see the
[2 November result](docs/results/2026/11-02/fixture-checks.md) for exact cases,
source guards and cleanup.

The separate [7 November public-file observations](docs/results/2026/11-07/public-gguf-observation-attempt-2.md) completed on
Linux/Python 3.12.3, using three fixed complete file hashes. The small container
has 48 F32 tensors; the 15M reference has 57 F32 tensors. Both were admitted.
The Q4-labelled file has 43 Q4_0, 13 F32 and one Q8_0 tensor; its real inspect
exit 2 is preserved because Q8_0 is outside this profile. The explicitly selected
288 × 288 reference matrix is F32, rather than required F16. The restoration
pair remains unadmitted; unquantized payload equality was not evaluated.
This is actual container evidence, with verified owned remote weight teardown,
not repair, inference, lineage or recovered behavior. No local weights are
included. The earlier create-event control remains UNRUN. The full project
remains unfinished at compatible-pair and behavioral acceptance gates.

This is a local CLI, so no Vercel/Supabase deployment is needed for its current
execution path. No account, paid plan or provider is provisioned.

## Opt-in compatibility v2

The separate [v2 contract](docs/proposals/opaque-q8-f32-v2.md) adds
`--profile v2` to `inspect`, `plan` and `repair`. Default v1 keeps the supported
profile and F16 restoration examples above. V2 additionally admits Q8_0 as
opaque unselected payloads and restores explicit Q4_0 matrices from matching
F32 references. It does not decode Q8 values. `diagnose` and `probe` remain v1.

```sh
python3 -m quantopsy inspect quantized-with-q8.gguf --profile v2
python3 -m quantopsy plan reference-f32.gguf quantized-with-q8.gguf --profile v2 \
  --tensor blk.0.attn_q.weight --provenance local-provenance.json \
  --max-added-bytes 67108864 --output repair-plan-v2.json
python3 -m quantopsy repair reference-f32.gguf quantized-with-q8.gguf --profile v2 \
  --plan repair-plan-v2.json --output-dir candidate-v2
```

These are usage examples. V2 records use schema version 2 and require the same
explicit profile when repaired. Both complete inputs, every unquantized binding,
selected payloads and the rebuilt output are rechecked; opaque Q8 payloads are
copied and hash-verified. Existing create-only and manifest-last rules apply.
The new source and its focused verification are pending; neither the historical
80-case result nor the public-file diagnostic proves this new path.

MIT licensed. Original implementation; see [primary sources](docs/sources.md).
