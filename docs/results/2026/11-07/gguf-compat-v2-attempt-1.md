# Opt-in v2 GGUF compatibility and candidate — attempt 1

Last updated: 2026-11-07.

**Verified bytes; product acceptance remains false.** First invocation:
92 fixture passes, both pinned 15M containers admitted under explicit
`--profile v2`, restoration-pair bindings verified and candidate created/reparsed.
`behavioralValidation: not_run`; no inference or recovered-behavior claim.

## Run and immutable originals

- [Run 42378284921 / job 128420323583](https://github.com/alvarovillalbaa/quantopsy/actions/runs/42378284921/job/128420323583).
- Checked [source e18dee0559949f0bbedba01da9c3006b228c44cd](https://github.com/alvarovillalbaa/quantopsy/tree/e18dee0559949f0bbedba01da9c3006b228c44cd),
  parent `3f3b3aedabb85e5909db6910df6c67a2d3873b18`;
  [QA activation 7b65df51](https://github.com/alvarovillalbaa/quantopsy/commit/7b65df51aad9c8ae3a7241203981ced604e60a64).
- Controller interval: 13:43:54.754–13:44:03.070 UTC. Workflow
  `410349161` was disabled at 13:44:23 UTC; fresh readback followed.
- [Original JSON](gguf-compat-v2-attempt-1.json): 447,480 bytes,
  SHA256 `34f6ed5e6348827617a364a1f6512e7f93ba0dedb126c02a9822b35b039e8bca`.
- [Original workflow log](gguf-compat-v2-attempt-1.log): 465,012 bytes,
  SHA256 `31d5b6d2cfa09dcaaae4366481133c6ba0b58d6fbe5209fea41b548c5ed542d1`.
  Saved before parsing; no later publication or deployment is established here.

## Checks

Preinstalled isolated Python 3.12.3 collected and ran
`tests.test_gguf` (26), `tests.test_repair` (22), `tests.test_cli` (16), and
`tests.integration.test_profile_v2` (28): **92 passed**, with no failures,
errors, skips, expected failures or unexpected successes. These are authored
Tier 2 fixtures; the unchanged 16-case diagnostic suite was not repeated.

Both public inputs were downloaded at revision
`4cd12190a40543c61bad59a1b8d982dfc48de020`, with complete received/written
bytes, full SHA256, body EOF and closed HTTP/file descriptors:

| Input | Bytes | SHA256 |
| --- | ---: | --- |
| stories15M.gguf | 98,357,920 | `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a` |
| stories15M-q4_0.gguf | 19,077,344 | `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04` |

Each followed one HTTPS 302 from Hugging Face to
`us.aws.cdn.hf.co`, then completed; signed query values were not retained.
The real CLI sequence was v2 inspect(reference/quantized), independent pair
binding, explicit plan, repair, v2 candidate inspect and independent candidate
binding. All **41 child commands exited zero** and closed normally.

The reference has 57 F32 tensors; its partner has 13 F32, 43 Q4_0 and one Q8_0.
Fresh hashes established all 13 unquantized bindings and metadata equality
outside the four documented quantization keys. The selected
`blk.0.attn_q.weight` is 288×288: 82,944 elements, 331,776 reference F32 bytes
replacing 46,656 Q4_0 bytes. Requested added-weight budget: 285,120 bytes;
writer reserve: 1,073,741,824 bytes.
Authored provenance matched plan/manifest exactly; its MIT/revision/source
declarations do not establish cryptographic lineage or a legal conclusion.

Candidate: **19,362,432 bytes**, SHA256
`b6e6456eab2d94c00d36f603949f796b215ea9055e8f90970fc92284a1ee35de`.
All 57 candidate payload bindings passed: the selected payload matches the
reference and all 56 unselected payloads match the quantized input.
The 9,792,000-byte `output.weight` Q8_0 payload remained opaque and unchanged.
Candidate: 14 F32 / 42 Q4_0 / 1 Q8_0. Metadata order/bytes were preserved except
`general.file_type`; bundled/external plans and manifest matched. Whole-file
size was measured, not inferred from the growth budget.

## Source, originals and teardown

Five guards covered all 49 files / 989,405 bytes before/after download, after
observations and before/after cleanup. Nine changed prior paths were pinned;
all 32 unmodified paths, including the inert baseline workflow, remained exact.
The JSON contains controller/helper/test hashes and the full source vector.

The report contains 82 command-stream gzip originals, five nested descriptor
originals and three JSON-file originals (90 typed gzip records). Nested copies
are not extra executions. Hashes and bounded structural values are retained;
weight payloads and raw GGUF metadata-region bytes are not retained for local
rehash.

All children were reaped, pipes reached EOF, descriptors closed and original
groups disappeared without TERM/KILL. Ten owned files (136,853,939 bytes)
passed identity/full-hash checks and were removed. Source-enforced inode checks
covered nine known directories; their raw inode values were not serialized.
Remaining and uncertain process inventories were empty. Original HOME was preserved.
Boundary checks are not continuous filesystem or host-wide attestations.

The [v2 contract](../../../proposals/opaque-q8-f32-v2.md) remains opt-in.
The [earlier v1 Q8/pair refusal](public-gguf-observation-attempt-2.md) and
[first create-control UNRUN](public-gguf-observation-attempt-1.md) remain
historical facts. Inference, quality, provider/device acceptance, Q8 numerical
decoding, selected-Q8 restoration and full-product completion remain untested.
No weights or candidate artifact are committed.
