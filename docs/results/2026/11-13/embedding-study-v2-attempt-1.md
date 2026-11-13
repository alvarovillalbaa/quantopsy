# Frozen embedding study v2 — completed, negative

13 November 2026. **Operational completion accepted; both recovery verdicts
NEGATIVE.** Four actual CPU role runs completed on eight predeclared authored
stories, with identical tokenization and 368 target tokens per role. Neither
candidate satisfies every unchanged recovery/regression requirement. Full
product acceptance and general behavioral recovery remain open.

| Candidate | Qualifying losses | Recovered | Excessive regression cases | Recovery pass | Actual bytes |
| --- | ---: | ---: | ---: | --- | ---: |
| v2, 42 internal matrices | 12 | 9 | 4 | false | 39,605,952 |
| v3, those matrices plus input embedding | 12 | 12 | 4 | false | 71,285,952 |

V3 recovered all qualifying losses, but regressed against the quantized baseline
on scarf/hat/bridge token accuracy and puzzle mean target NLL. The frozen rule
forbids excessive regressions even on cases without a qualifying loss. V2's four
regression cases and both complete comparison reports remain in the originals.
This is a narrow result for these tasks, model and settings, not proof of general
recovery, an optimal quantizer, or a causal explanation of the old experiment.
The old study's 11/10/four negative verdict is unchanged; scores on different
task sets are not an improvement comparison.

## Source and originals

- Prepared source [`4f06e453b64de1ebcae87ed6d4407989e4ba91a0`](https://github.com/alvarovillalbaa/quantopsy/commit/4f06e453b64de1ebcae87ed6d4407989e4ba91a0), tree
  `0402389bdb606988724599147d880ccb621b83cc`: 88 files / 3,799,446 bytes.
- [Actual run 43394377250](https://github.com/alvarovillalbaa/quantopsy/actions/runs/43394377250), job
  `131781628289`, attempt 1, terminal `success`. Activation commit
  `4ce9057335f5f634c17bbfe928fcde086f73a67e` adds only the exact active workflow.
  A green CI execution is separate from the negative measurement decisions.
- [Original artifact ZIP](embedding-study-v2-attempt-1.zip), artifact
  `14021394647`: 205,725 bytes, SHA-256
  `be42772ec90c99a7ceae5c7ce075a13a8ebaf2be44e221802e19ac9d56399a23`.
  Its sole original log member is retained losslessly inside the ZIP.
- [Exact extracted report](embedding-study-v2-attempt-1.json): 603,795 bytes,
  SHA-256 `e57e51ebc0d03e1695c3a1aa932fae3208489eeaaccbcfdfd915e74b59b530d7`.
  The ZIP member equals the result marker, these exact JSON bytes and a newline.
  The report retains every original command stream, frozen input, run,
  comparison, failure list and cleanup observation.
- The [predeclared protocol](../../../studies/embedding-v2.json) remains byte-exact,
  SHA-256 `8c35578fb0f8a2451a876f0a86cacf98941f7887a289df0bfd19c5ae416c5bc3`.
  Its `unrun` field describes its state when frozen, not the current outcome.

The retained [74-case qualification](profile-v3-checks.md) was carried through
exact source/case/two-literal checks; no fixture case reran. All product code,
tests, old study, scorer, CMake recipe and frozen JSON stayed exact. This
publication adds evidence and current-status documentation only.

## Actual models and engine

The pinned public reference/quantized files were freshly downloaded, fully
hashed and independently inventoried. All 57 candidate payloads were checked
for each profile, including the unchanged opaque Q8 output. No model, binary,
full tokenizer payload, checkpoint, provider call or private data is published.

| Role | Bytes | SHA-256 |
| --- | ---: | --- |
| F32 reference | 98,357,920 | `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a` |
| Quantized | 19,077,344 | `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04` |
| Fresh v2 candidate | 39,605,952 | `a0107dd0112554eeaa648c3f64e6017e61f1531327e993191c8bea7326b585da` |
| Fresh v3 candidate | 71,285,952 | `16157d19ac9c84fdfc5ece112283b362600cf8324a45e2e8b5be737e0d007617` |

Official llama.cpp `74d4f5b041ad837153b0e90fc864b8290e01d8d5` was reconstructed
against full Git tree `9ae6f9dd0984b780c942ccf510e9cd5fcee9e724`, with its
declared disabled/unfetched gitlink. The five-file engine-source manifest binds
the unchanged scorer/build recipe, new helper/controller and protocol. Actual
2,629,016-byte scorer binary SHA-256 is
`56e8dcd1cd56f5def7f7ab21cbddc06e51998d4c696751e2bc9ca67b17c876a1`.

Four actual teacher-forced runs used two CPU threads, context/batch 128, BOS,
no EOS/sampling, F32 KV and the frozen engine settings. Observed forward times
were reference 449 ms, quantized 338 ms, v2 388 ms and v3 409 ms. These are
single-host observations, not portable performance benchmarks. V3 adds
52,208,640 weight bytes, versus 20,528,640 for v2; its remaining compression
tradeoff is material despite its better loss-recovery count on this task set.

## Independent audit and closure

The [data-only verifier](verify-embedding-study-v2.py) and [passing audit 4](embedding-study-v2-audit-4.json)
check all 145 compressed originals, 124 complete streams / 62 normally closed
commands, all 88 source Git blobs, seven unchanged full source/index guards,
the exact task wire/thresholds, four-role token coverage, 64 independently
aggregated metric values, both 16-case decisions, original plan/manifest/run
hashes and reported owned teardown. No local model/build/fixture replay occurred.

Three local audit-observer failures are preserved in [audit 1](embedding-study-v2-audit-1.json),
[audit 2](embedding-study-v2-audit-2.json) and [audit 3](embedding-study-v2-audit-3.json).
They incorrectly required command-only flags on inventory captures, reserialized
product JSON, and compared a document wrapper to a role map. The two later
failed verifier versions are archived as inactive text. Audit 4 corrects these
observer assumptions using original bytes; no runner result, task or threshold
changed and no inference reran.

The runner verified and removed 1,452 owned files / 222 directories, with owned
file-vector SHA-256 `ea5cd7a8772be580f59e901d689a6f1980fec2f94df35dc3d59c1416a8a5968b`.
All original groups were observed absent, all child leaders reaped, descriptors
closed and pipes at EOF; no signals, uncertain or remaining owned items were
reported. Scratch cleanup passed. These are scoped observations, not an
all-host disposal or external host-attestation claim. Imported records do not
authenticate execution or cryptographic model lineage by themselves.

The [retirement receipt](embedding-study-v2-retirement.json) records workflow
`417050024` disabled and the SHA-verified QA ref deleted, with empty ref readback.
The small local transfer ZIP and redundant extracted log were removed after
identity/hash/no-openers checks; the canonical ZIP and raw JSON remain. Local
Storage Guard still forbids heavy work. This experiment is closed: preserve
the negative result without rerunning or adjusting it to obtain a pass.
