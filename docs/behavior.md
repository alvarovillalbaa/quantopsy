# Data-only behavioral records

`python3 -m quantopsy compare-behavior` compares supplied measurements without
inference, networking or reading GGUF files. Inputs are bounded regular
non-symlink JSON, strict UTF-8, no duplicate keys or nonfinite numbers and exact
schemas rejecting extra fields. Hashes cover the same bytes actually parsed.

Supply `--suite`, `--config`, `--models`, `--reference-run`, `--quantized-run`,
`--candidate-run`, optional `--repair-manifest`, and new `--output`. Successful
report creation returns 0 even when criteria fail. Invalid records/refused
outputs return 2 with finite structured errors. Existing paths are preserved.

Every record has integer `schema_version: 1` and exact `kind`:

| Kind | Remaining fields |
| --- | --- |
| `quantopsy.behavior-config` | `engine`, `tokenizer_sha256`, `context_tokens`, `batch_tokens`, `threads`, `seed`, `add_bos`, `add_eos` |
| `quantopsy.behavior-suite` | `suite_id`, `config_sha256`, ordered `metrics`, ordered `tasks` |
| `quantopsy.behavior-models` | `reference`, `quantized`, `candidate` |
| `quantopsy.behavior-run` | `role`, `suite_sha256`, `config_sha256`, `engine`, `model`, `status`, `error`, ordered `tasks` |

Engine: `{name, revision, source_sha256}`. Model: `{name, sha256, size}`,
a filename label, full lowercase SHA-256 and bounded positive byte count. Each
run binds its explicit role's model, identical engine and exact suite/config
bytes including whitespace. Identical models may be controls; equality cannot
prove quantization loss.

A suite has 1–128 unique tasks `{id, prompt, target}` and 1–8 unique metrics
`{id, direction, min_quantization_loss, min_recovery_fraction,
max_reference_regression, max_quantized_regression}`. Direction is `lower` or
`higher`. Measurements/thresholds are finite bounded decimal **strings** without
exponents. Loss/regression bounds are nonnegative; fractions are in [0,1].

Run task: `{id, metrics: [{id, value}, ...]}` in exact suite task/metric order.
`complete` requires all tasks and `error: null`. `incomplete`/`error` requires
a bounded nonempty error and permits an ordered completed prefix. Missing values
remain null/unavailable and force a negative overall verdict.

Orient differences so positive means worse. Qualifying loss must be positive
and meet its minimum. Recovery needs positive improvement at least loss ×
minimum fraction. Exact Decimal products determine acceptance; displayed
fractions do not. Any candidate regression beyond either bound fails, even
without qualifying loss. Passing requires complete runs, at least one loss,
all qualifying losses recovered and no excessive regression.

Optional repair manifests must satisfy the existing strict plan contract,
bind all models and declare all byte checks complete. This checks records,
not actual payloads. Reports disclaim inference execution, model opening,
freeze chronology and execution authenticity. `supplied_recovery_records_pass`
describes records only. Independently verified token/execution evidence is
needed for an experimental claim; broader claims need broader studies.
