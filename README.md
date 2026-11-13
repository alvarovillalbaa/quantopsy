# Quantopsy

Investigate quantization loss and create a local weight-restoration candidate.

The source inspects GGUF containers, compares selected weights and supplied
linear activation probes, and restores explicitly selected Q4_0 tensors from
an F16 reference (default v1) or F32 reference (explicit v2/v3) into a NEW GGUF bundle.
V3 additionally permits explicit restoration of the Q4_0 input embedding after
checking its width and tokenizer vocabulary shape.
It uses Python 3.11+ and the standard
library on POSIX macOS/Linux, with no model download, hosted service or inference
engine. No Python/platform runtime matrix has been verified yet.

**The first inference experiment failed its recovery criteria.** Actual
reference/quantized/candidate inference recovered 10 of 11 qualifying losses
but exceeded four regression limits on eight frozen story tasks. The
[full result](docs/results/2026/11-08/behavioral-inference-attempt-2.md) retains
the negative verdict and original token measurements. Parameter error and
linear probes remain separate diagnostics; general behavioral recovery and
the [full product objective](docs/spec.md) remain open.

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

## Compare supplied task measurements

`compare-behavior` checks frozen reference/quantized/candidate records against
explicit loss, recovery and regression limits. Missing/failed tasks stay visible;
mismatched identities are refused. It does not execute models or authenticate
imported results; see the [record contract](docs/behavior.md).

```sh
python3 -m quantopsy compare-behavior --suite suite.json --config config.json \
  --models models.json --reference-run reference-run.json \
  --quantized-run quantized-run.json --candidate-run candidate-run.json \
  --repair-manifest candidate-bundle/repair.json --output comparison.json
```

This is a usage example. The [first frozen experiment](docs/results/2026/11-08/behavioral-inference-attempt-2.md)
completed real CPU inference and verified teardown, with a negative recovery
decision. The application itself still imports data-only measurements.

The [first invocation](docs/results/2026/11-08/behavioral-inference-attempt-1.md)
passed 51 focused checks and built the scorer, then failed a build-file admission
guard before inference. Its cleanup remains unverified. The retained failure
does not establish recovery; the changed control keeps the frozen study intact.

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
remained unfinished at compatible-pair and behavioral acceptance gates in that
v1 observation. The separate v2 result below establishes candidate-byte checks.

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
The [first v2 result](docs/results/2026/11-07/gguf-compat-v2-attempt-1.md)
passed 28 new cases and 64 relevant v1 regressions at `e18dee05`, then created
and independently verified a candidate from the two pinned public 15M files.
All 13 unquantized bindings and 57 prescribed output payloads matched, including
the unchanged opaque Q8 payload. The 19,362,432-byte candidate was disposed of
after verification; it is not distributed here. The earlier v1 refusal remains
unchanged. Model inference, recovered behavior and full-product acceptance
were not evaluated by that byte-level packet. The separate
[8 November experiment](docs/results/2026/11-08/behavioral-inference-attempt-2.md)
completed inference and failed its recovery criteria. Default v1 still excludes
Q8 and requires selected F16 bytes.

MIT licensed. Original implementation; see [primary sources](docs/sources.md).

## Explicit input-embedding restoration v3

Use `--profile v3` on each inspect/plan/repair operation to allow the additional
target `token_embd.weight`. Its F32 reference and Q4_0 payload must match the
declared Llama width and tokenizer count; optional vocabulary-size metadata
must agree. Selection remains explicit. V1/v2 defaults, targets and record
versions are unchanged; v3 plans/manifests use version 3. The Q8 output head
remains unselected/opaque. No tied weights are inferred or restored implicitly.

```sh
python3 -m quantopsy plan reference-f32.gguf quantized-with-q8.gguf --profile v3 \
  --tensor token_embd.weight --provenance local-provenance.json \
  --max-added-bytes 67108864 --output repair-plan-v3.json
python3 -m quantopsy repair reference-f32.gguf quantized-with-q8.gguf --profile v3 \
  --plan repair-plan-v3.json --output-dir candidate-v3
```

These are usage examples. [Seventy-four authored fixture identities](docs/results/2026/11-13/profile-v3-checks.md)
passed as 72 first-run passes plus two corrected refusal expectations, without
replaying the 72. Original failures, the observer permission error, explicit
public-path redactions and scoped cleanup are retained. No new model was run.
The [frozen follow-up study](docs/studies/embedding-v2.md) compares a legacy
42-matrix candidate and a 43-tensor embedding candidate on eight new tasks with
unchanged thresholds. Its exact CI controller and real inference remain unrun;
the previous negative result is preserved. Restoring more weights costs space
and does not establish optimal quantization or general recovery.
