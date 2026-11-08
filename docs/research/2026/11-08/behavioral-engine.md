# Frozen public 15M CPU inference protocol v1

Prepared 8 November 2026. This document defines a bounded experiment before
engine execution; it contains no passing result. The owning controller records
actual build, inference, failures and cleanup separately. No local model,
application runtime, test, download, installation or build is authorized by this
document; the local storage reserve remains unsatisfied.

The decision is whether one explicitly declared restoration improves the
quantized model's predictions on eight fixed story continuations while meeting
the declared regression limits. Actual neural forward passes are required.
Weight hashes, fixture passes, a successfully created GGUF and caller-supplied
JSON cannot establish that decision.

## Primary source inspection and engine choice

The official llama.cpp `b5046` release resolves to immutable commit
`74d4f5b041ad837153b0e90fc864b8290e01d8d5`. The exact source inspected was its
[C API header](https://github.com/ggml-org/llama.cpp/blob/74d4f5b041ad837153b0e90fc864b8290e01d8d5/include/llama.h),
[simple decoder example](https://github.com/ggml-org/llama.cpp/blob/74d4f5b041ad837153b0e90fc864b8290e01d8d5/examples/simple/simple.cpp),
[perplexity implementation](https://github.com/ggml-org/llama.cpp/blob/74d4f5b041ad837153b0e90fc864b8290e01d8d5/examples/perplexity/perplexity.cpp),
[argument definitions](https://github.com/ggml-org/llama.cpp/blob/74d4f5b041ad837153b0e90fc864b8290e01d8d5/common/arg.cpp),
[CPU build instructions](https://github.com/ggml-org/llama.cpp/blob/74d4f5b041ad837153b0e90fc864b8290e01d8d5/docs/build.md)
and [backend build options](https://github.com/ggml-org/llama.cpp/blob/74d4f5b041ad837153b0e90fc864b8290e01d8d5/ggml/CMakeLists.txt).

That release's `llama-perplexity` accepts `--ctx-size`, `--batch-size`,
`--ubatch-size`, `--threads`, `--threads-batch`, `--chunks`, `--ppl-stride` and
`--ppl-output-type`. Its default non-strided calculation requires at least twice
the context length in tokens, scores only the second half of each full chunk
apart from the final next-token position, and ignores incomplete trailing
chunks. Output type 1 additionally prints a rounded mean NLL. Some early score
failures return a failed result internally while `main` still returns zero.
An exit code alone therefore cannot establish a measurement. Those chunk
semantics do not directly score these named prompt/target boundaries.

The separate authored [C++ scorer](../../../../tests/helpers/behavioral_scorer.cpp)
uses the pinned API to expose exact target scores. The application does not
import it, download it, call subprocesses or load an engine. The
[trusted data helper](../../../../tests/helpers/behavioral_inference.py) uses
only Python's standard library and performs no subprocess invocation or model
execution. The [CMake wrapper](../../../../tests/helpers/behavioral_engine/CMakeLists.txt)
links one scorer target against the already verified source directory. It
disables common tools, tests, examples, server, CURL, external backends, dynamic
backend loading, native CPU tuning, OpenMP, BLAS and optional downloaded CPU
kernels. The controller permits two build jobs and uses preinstalled tools;
there is no package installation or hosted inference service.

## Frozen inputs, task data and candidate

Use exactly the same two immutable public inputs from the
[7 November v2 receipt](../../../results/2026/11-07/gguf-compat-v2-attempt-1.md),
Hugging Face revision `4cd12190a40543c61bad59a1b8d982dfc48de020`:

| Role | File bytes | SHA256 |
| --- | ---: | --- |
| Reference | 98,357,920 | `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a` |
| Quantized | 19,077,344 | `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04` |

The saved independent inventory reports `llama.context_length=128`. Use context
128 and batch 128, two inference threads, FP32 K/V cache, CPU only, no Flash
Attention and no sampling. Seed zero is recorded but unused. Refuse context
larger than the model's actual training-context metadata. Refuse token overflow
instead of truncating, padding, splitting or changing the tasks.

The eight short ordinary English prompt/target stories are authored in `TASKS`
in the trusted helper. They cover a boat, seed, key, rain, sharing, bird, cake
and room light. They were created independently for this experiment, before
any observed model output, and are not copied from a benchmark. They are held
out from candidate selection; complete disjointness from unknown upstream
training data is not established. Do not edit their text or thresholds in
response to inference outcomes.

The single new candidate explicitly restores all seven supported dense matrices
in each of six blocks: `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`,
`ffn_up` and `ffn_down`, each with the `.weight` suffix. The fixed 42-name list
is exposed as `SELECTED_TENSORS`; selection uses no score, ranking or search.
The saved inventory implies 5,971,968 selected elements, 23,887,872 F32 bytes
replacing 3,359,232 Q4_0 bytes, and exactly 20,528,640 added weight bytes.
Fresh pair, plan and payload checks must confirm these values remotely.
Q4 token embeddings, opaque Q8 output weights and every other unselected
payload remain byte exact. This deliberately broad mixed-precision restoration
does not establish a compact optimal repair. The prior single-matrix candidate
and its disposed artifact remain a separate historical byte-level observation.

## Exact scoring and comparability

Tokenize the prompt and its concatenation with the target using the same model
tokenizer. Set `parse_special=false`. Require actual tokenizer BOS addition
and no EOS addition; require the separately tokenized prompt to be an exact
token prefix of the full text. Check the actual first token is BOS. Refuse
boundary mismatches. No EOS or prompt token is scored.

For every target position, evaluate its preceding prompt and ground-truth
target prefix. Request that preceding position's logits, check every vocabulary
logit is finite, and calculate stable log-softmax with a serial double-precision
sum over the entire vocabulary. Preserve each target NLL as a fixed 17-place
decimal string. Argmax chooses the lowest token ID in an exact tie. Preserve
the full token sequence, actual target IDs and actual argmax IDs. Clear the KV
cache between tasks; require exactly one score per target token. Each model is
loaded once for all eight tasks.

The Python helper independently recomputes mean target NLL and exact target
token accuracy from those originals with Decimal arithmetic, then rounds once
to 12 decimal places using `ROUND_HALF_EVEN`. This aggregation does not
independently recompute logits. Target accuracy is teacher-forced next-token
accuracy; it is not free-generation accuracy or a semantic correctness grade.

Before scoring, fingerprint the tokenizer metadata by sorting all `tokenizer.*`
entries by key and hashing canonical JSON records containing `key`, `kind`,
`arrayKind`, `count`, `encodedBytes` and `encodedSha256`. Require identical
signatures across the three actual GGUF files. After scoring, require identical
prompt counts, target counts, full token IDs, target IDs and vocabulary size
across roles. Token equality is a comparability guard, never a recovery score.

| Metric | Direction | Minimum quantization loss | Minimum recovered fraction | Maximum regression from reference | Maximum regression from quantized |
| --- | --- | ---: | ---: | ---: | ---: |
| `mean_target_nll` | lower | 0.005 | 0.5 | 0.02 | 0.01 |
| `target_token_accuracy` | higher | 0.02 | 0.5 | 0.03 | 0.02 |

These are predeclared engineering thresholds for this small experiment, not
population-calibrated scientific standards. The strict comparator requires
complete runs, at least one qualifying loss, recovery of every qualifying
task/metric case and no exceeded regression limit. Better quantized performance
or an absence of qualifying losses cannot become a recovery claim. A failed,
inconclusive or negative outcome remains visible and is not retuned into a pass.

## Source binding, original evidence and resource boundary

`engine_manifest(repo)` binds the exact authored C++ scorer, Python aggregation
helper and CMake wrapper with relative paths, sizes and full SHA256, the pinned
upstream commit and the fixed recipe. Hash its canonical raw bytes for
`engine.source_sha256`. The immutable revision string is
`74d4f5b041ad837153b0e90fc864b8290e01d8d5+quantopsy-target-nll-v1`.
The controller separately records the actual complete downloaded source archive
SHA256, extracted source inventory, compiler/build output, build configuration
and built binary SHA256. No archive checksum or successful build is asserted
here: source-archive hashing was unavailable locally after DNS lookup failed.

The official immutable commit's Git tree was independently read through the
GitHub API as `9ae6f9dd0984b780c942ccf510e9cd5fcee9e724`. The engine manifest
binds this value as `upstream.git_tree_sha1`. Before compilation, the controller
must reconstruct the complete downloaded source tree with Git's blob/tree
encoding and compare its root to that identity. The archive's actual SHA256
remains a separate observed transport receipt until CI produces it.
The official recursive tree contains 1,240 blobs totaling 68,926,775 bytes and
one Git submodule entry: `ggml/src/ggml-kompute/kompute`, mode `160000`, commit
`4565194ed7c32d1d2efa32ceab4d3c6cae006306`. Source archives omit that submodule's
payload. The manifest pins the omitted Git entry so tree reconstruction can
include its identity while requiring the archive path to be absent or empty.
Do not fetch that submodule; its Kompute backend stays disabled. No symbolic
links were present in the official recursively returned tree, which was not
truncated.

The controller writes and hashes the engine manifest, exact config, ordered
suite, TSV task bytes, model bindings and repair manifest before the first
forward pass. Model-role run records bind the raw suite/config hashes, engine
identity and full model identity. Keep the exact engine stdout/stderr originals
and their hashes, including per-token evidence; strict run DTOs contain only
task metrics. Bind all other source files through the controller's complete
checked-source vector. Verify source, engine, tokenizer and model identities
again after inference.

Execution belongs to one owned free public Ubuntu CI packet, at most 20 minutes
including build, model handling, measurement and cleanup. The owning controller
enforces storage, memory, process/output limits, retained original streams,
normal child closure and verified removal of its own weights, candidate, engine
source/build and temporary task files. No weight or tokenizer payload is
committed; only authored tasks, source and bounded evidence are retained. A
process/build/scoring/cleanup failure blocks the verified experiment result.

The [TinyStories paper](https://arxiv.org/abs/2305.07759) motivates short simple
story language for small models. The
[Hugging Face perplexity definition](https://huggingface.co/docs/transformers/perplexity)
supports conditional token likelihood as a language-model prediction measure
and notes its tokenizer dependence. This study measures prediction loss on
eight authored continuations. It cannot establish general semantic recovery,
free-generation quality, reasoning, safety, unseen-distribution performance,
other models/quantizers or full Quantopsy product acceptance.
