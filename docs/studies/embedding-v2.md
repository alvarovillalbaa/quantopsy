# Predeclared input-embedding follow-up, behavioral study v2

13 November 2026. **State at predeclaration: UNRUN.** The subsequent
[actual result](../results/2026/11-13/embedding-study-v2-attempt-1.md) completed
four role runs and is negative for both candidates: v2 12 qualifying / nine
recovered / four regressions; v3 12 / 12 / four. Protocol JSON remains byte-exact.
Preserve the [first study's negative result](../results/2026/11-08/behavioral-inference-attempt-2.md):
11 qualifying losses, 10 recovered, four regression cases. The old eight tasks,
thresholds, candidates, scorer and originals are immutable. The Notion brief
does not guarantee a positive recovery result.

The retained model inventory shows a Q4_0 input embedding left unchanged by
the first 42-matrix candidate. The Q8_0 output head is different and stays opaque.
This supports testing a hypothesis; it does not prove a cause of the first
study's failures. V3 makes explicit input-embedding restoration available while
retaining older profile semantics.

The [immutable data contract](embedding-v2.json) freezes eight new authored
continuations, their order, role/candidate strategy, unchanged thresholds and
the pinned model/engine identities before any new model output. None is an old
prompt or target. This is a versioned follow-up informed by the previous result,
not a blind independent replication or an outcome-tuned revision of study v1.

Measure four fresh model roles on the same new task bytes:

1. The pinned F32 reference.
2. The pinned quantized model.
3. A freshly rebuilt legacy v2 candidate restoring the same 42 internal matrices.
4. One v3 candidate restoring those matrices plus `token_embd.weight`.

Use the same pinned official llama.cpp/scorer, CPU two-thread configuration,
context/batch 128, explicit BOS/no EOS, no sampling, and teacher-forced original
token measurements. No task, threshold or strategy may change after output.
Freeze both fresh candidate identities, engine binary, suite, tokenizer and
models before the first forward. Verify the old 42-matrix candidate's exact
published identity. Independently verify every input and every candidate payload.

Compare each candidate to the shared reference/quantized measurements through
its separately bound triplet. Retain all four raw runs, both comparison reports,
failures, regressions and token denominators. Require identical tokenization.
The v3 candidate must pass every unchanged recovery/regression rule; zero
qualifying losses are inconclusive. Report whether the legacy candidate passes
or fails separately. Passing both does not prove the extra embedding cost was
necessary. Do not claim an improvement by comparing old tasks to new tasks.

Predeclare the tradeoff: v3 selects 15,187,968 elements and adds 52,208,640 weight
bytes, versus 5,971,968 / 20,528,640 for v2. Expected rebuilt sizes are 71,285,952
and 39,605,952 bytes respectively, compared with 98,357,920 reference and
19,077,344 quantized bytes. Report actual exact sizes and runtime observations;
this approach preserves less compression and is not an optimal quantizer.

One new free public Ubuntu job, capped at 20 minutes, may build the pinned
official engine and execute this frozen protocol after its exact controller is
reviewed and published under the owner's standing all-work authority. No
application dependency, model API, account, payment, private data, local model
execution or provider setting is involved. All weights/builds stay in bounded
owned runner storage with original streams, exact source/input guards, process
closure and teardown. Do not activate an incomplete controller or replay the
closed fixture scopes as a substitute for this experiment.

Fixture acceptance can qualify v3's file/record behavior only. This tiny
teacher-forced story suite, even if positive, cannot establish general semantic
recovery, other models/quantizers, free generation, optimal selection or the
full product. Preserve a negative or inconclusive result without retuning.
