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
[delivery receipt](docs/results/2026/10-29/initial-source.md). Domain checks are
prepared separately and remain unexecuted until explicitly approved. No real
checkpoint, model inference or behavioral-quality result is claimed. The full
project remains unfinished at those acceptance gates.

This is a local CLI, so no Vercel/Supabase deployment is needed for its current
execution path. No account, paid plan or provider is provisioned.

MIT licensed. Original implementation; see [primary sources](docs/sources.md).
