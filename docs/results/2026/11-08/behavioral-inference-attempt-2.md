# Completed inference experiment — recovery criteria failed

8 November 2026, 07:48:53–07:50:06 UTC. [Run 42460300956](https://github.com/alvarovillalbaa/quantopsy/actions/runs/42460300956),
job `128667959481`, completed at checked source
`40b0e846aef56c9e950823976c6a39d7c22c233c`, workflow
`16ff58c26d0b18053c189d912e24ec0cbd71f81f`. Actual CPU inference covered
reference, quantized and repaired models on all eight frozen tasks. The
experiment completed; **the candidate failed the frozen recovery criteria**.

There were **11 qualifying losses, 10 recovered cases and four regressions**.
NLL: six qualifying/six recovered/three regressions. Accuracy: five qualifying/
four recovered/one regression. Complete model coverage and a verified repair
manifest do not turn these results into acceptance. No tasks, thresholds or
candidate selection were changed after model output.

## Actual measurements

Each role scored **382 target tokens** plus 97 prompt/BOS tokens across eight
tasks. Target counts were 44, 44, 47, 46, 48, 51, 53 and 49; complete task
lengths were 55–66 tokens, within the fixed context 128. Token IDs and boundaries
matched across roles, with BOS ID 1 and no EOS addition. These are teacher-forced
next-token predictions, not free-generation or semantic grades.

Token-weighted summaries below are descriptive. Acceptance uses the frozen
per-task/metric thresholds rather than these aggregate values.

| Model | Mean target NLL | Correct target tokens | Target accuracy |
| --- | ---: | ---: | ---: |
| Reference F32 | 2.237004967867 | 172/382 | 0.450261780105 |
| Quantized | 2.326591125004 | 163/382 | 0.426701570681 |
| Repaired candidate | 2.247128547845 | 176/382 | 0.460732984293 |

All failures under the predeclared criteria:

| Task / metric | Reference | Quantized | Candidate | Failure |
| --- | ---: | ---: | ---: | --- |
| Boat / NLL | 2.059566280574 | 2.160486612320 | 2.108587817294 | Reference regression 0.049021536720 exceeds 0.02 |
| Boat / accuracy | 0.431818181818 | 0.477272727273 | 0.431818181818 | Quantized regression 0.045454545455 exceeds 0.02 |
| Key / NLL | 2.525355149696 | 2.443143785626 | 2.503035784845 | Quantized regression 0.059891999219 exceeds 0.01 |
| Key / accuracy | 0.446808510638 | 0.425531914894 | 0.425531914894 | Zero improvement; minimum required 0.010638297872 |
| Rain / NLL | 2.048820856186 | 2.260467507547 | 2.110070860001 | Reference regression 0.061250003815 exceeds 0.02 |

Boat/rain NLL met fractional recovery while exceeding the separate reference
regression limit. Boat accuracy and key NLL require regression protection even
though quantization had improved that particular reference measurement. Key
accuracy retains one lost correct token. The [protocol](../../../proposals/behavioral-inference-v1.md)
and complete receipt preserve every other task/metric case.

## Frozen execution and candidate bindings

The same two pinned public inputs were freshly downloaded and hash-verified.
The explicitly selected 42 matrices and all 57 candidate payload bindings were
verified again: 42 exact reference payloads, 15 exact quantized-input payloads,
unchanged metadata/order except the declared file-type omission. Candidate:
**39,605,952 bytes**, SHA-256
`a0107dd0112554eeaa648c3f64e6017e61f1531327e993191c8bea7326b585da`,
identical to the first attempt. Selected payload growth remains 20,528,640 bytes.

Official llama.cpp b5046 source reconstructed Git tree
`9ae6f9dd0984b780c942ccf510e9cd5fcee9e724` from all 1,240 written blobs and
the pinned unfetched gitlink. The 20,880,037-byte archive again matched observed
SHA `8dd9fcc7c17f972673960ffdcc7521f52de5a9ec594aa6acebd9dfd8c17c0a5c`.
Build logs compile and link the reviewed scorer. All three forward commands
use one frozen **2,629,016-byte** binary, SHA-256
`4cf271e5f5fa97d5176944c7e5d1d3969e07121ff11b87d976c746da70ac1fc5`.
Hash-verified engine stderr reports context/training context 128, CPU/no GPU
offload, six layers and F32 K/V cache for each role.

The controller froze these exact originals at **07:49:59.364908 UTC**, before
the first forward, and checked unchanged files before/after every role:

| Record | SHA-256 |
| --- | --- |
| Engine source manifest | `26330dea1b224adfba5d510d91589b2a32f7eec4d06dd74606788b7916dcc7c1` |
| Configuration | `98565e1d8925f692859c6f29322388d1b7c9ae54aa11786810926724cc342c30` |
| Suite | `e63205e970b1269f227bec0a52fe9ce53a92d11deeed25328d5956d7406cca5f` |
| Tokenizer metadata fingerprint | `d6380a14d8f8d0937ff914927237c296412c8146ee5d4dec50781b8189e58aad` |
| Comparison report | `70e7b284bd57142f1fb27939c6ef77abbf17fdeeec750a9cb17202c7af22a9eb` |

The source manifest binds scorer, Python aggregation helper, CMake recipe and
upstream revision/tree. The receipt also retains exact models/TSV/run originals,
actual model identities, raw scorer streams, token IDs and per-token NLL strings.

## Checks, audit and teardown

The first invocation's **51 passing checks** were not rerun. This controller
bound their immutable receipt and all unchanged application/test/helper/scorer
inputs, explicitly recording `executed: false`. Its changed build-only admission
was exercised by the fresh build, inference and cleanup. The first failed
invocation and its unverified cleanup remain unchanged.

All **50 child commands** exited zero and closed normally. Six complete source
guards matched 66 files, 2,663,215 bytes and vector SHA
`f0cdcad15b82b4983de24efcd573aea5400d8414b8c4996266de07894fdb0d93`.
All original process groups were absent, children reaped, no signals or unknown
processes remained. Owned teardown verified **1,443 files and 221 directories**
and removed this invocation's scratch. Maximum sampled logical storage was
261,559,630 bytes, extracted source 68,926,775 bytes and build 14,549,230 bytes.
No local model, binary, upstream tree, build or app/test execution occurred.

Independent evidence-only reviews validated all **114 direct gzip captures**
(369,520 decoded bytes), original/parsed JSON parity, native Git/source bindings,
frozen records, all 24 task/role metrics and all 16 threshold decisions. Reviewers
used independent Decimal arithmetic over retained token NLL/target/predicted IDs,
without importing the application/helper or rerunning inference. This verifies
arithmetic and coverage from the trusted scorer; logits and disposed binary
payloads were not independently reconstructed or rehashed after teardown.

The workflow was disabled, and QA2 retired at
`d0a79daa52409485f7834b028baf43f3ed25eaf8`. Final publication removes the inert
default registration file. No account, paid API, provider, production data or
hosting service was introduced. The local Storage Guard restriction remains.

Immutable originals:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| [Receipt](behavioral-inference-attempt-2.json) | 410,559 | `8872d857e80abb764bb787c0225bba09e3ae1ce20679910bfd76224745be2ba3` |
| [Full job log](behavioral-inference-attempt-2.log) | 428,076 | `275f03ee0f3a0320a41f6f96d21635ca509817f5765d6a58fb78075d69dcc19f` |

This experiment establishes actual prediction measurements on eight authored
story continuations and a negative decision for one broad mixed-precision
candidate. It does not establish general semantic recovery, unseen-distribution
performance, other models/quantizers, optimal compact repair, model lineage or
full Quantopsy acceptance. Preserve this study; any further strategy needs a
separate versioned, predeclared evaluation with held-out confirmation.
