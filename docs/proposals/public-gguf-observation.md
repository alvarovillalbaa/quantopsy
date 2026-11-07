# Dispatch-only public GGUF observation — candidate 2

Status: preparation, unrun. The [first control](../results/2026/11-07/public-gguf-observation-attempt-1.md)
remains UNRUN, with zero observed workflow runs and no model/CLI result.
GitHub's [create-event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#create)
does not establish the cause. Do not attribute it to an unverified default-branch
rule, credential provenance or a runtime defect.

The [documented manual workflow path](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
requires a workflow_dispatch file in the default branch and permits selecting
another branch. Default base `4f2266434fd4e2c283ff0beddc12b4afece3e3f9`,
tree `b2352ea57828f2da4682fed0a121cd7ee36896b6`, contains 33 blobs /
245,928 bytes: all 31 prior inputs plus the 418-byte inert registration stub
and unchanged 3,195-byte UNRUN note. Stub source is
`.github/workflows/public-gguf-observation-2.yml`, SHA256
`a708a98e4def846cf3824373164dd48e01d127079bf9646a1fb2b2dee957731d`;
its only job has a constant-false condition. Registration was observed active
as workflow 410321779 at 12:35:54 UTC; no dispatch or resource job had run.

Candidate 2 changes only control fields: approved event workflow_dispatch,
exact full `refs/heads/qa/quantopsy-artifacts-20261107-2`, public repository,
fixed SOURCE_COMMIT and exact inert-stub source binding. If payload `ref`
is present, its short/full QA2 match is a defensive policy, not a claimed
documented payload-shape guarantee. No inputs expose code/source arguments.
The selected QA branch's active workflow supplies the guarded job; the
checked preparation retains the inert baseline body.

Use the [dispatch API](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event)
once with explicit QA2 ref only after registration and source review.
Record the configured API version's actual response rather than assuming
a run ID or status shape. Discover the returned/exact branch-head run using
workflow_dispatch, repository/ref/source bindings. Do not delete/recreate
the first branch, enable settings, change auth or silently retry a request.

The complete immutable file identities, independent observer, actual inspect
and pair-prerequisite logic, standard-library runtime, Tier 2 fixture scope,
512 MiB limit and 15-minute/600+90+30 budgets below remain unchanged.
No real-file or pair result is pre-authored. Five new packet paths plus scoped
AGENTS append on the 33-file base make 38 checked source paths; the existing
inert stub and first UNRUN note stay byte-exact. The old source and branch
are preserved; old QA was retired by removing only its active file.

The original artifact/resource contract follows as historical candidate-1
preparation. Its event/ref/base counts are superseded only by this control
edition; no runtime or domain claim is carried forward.

---

# Public GGUF container observations

Status: preparation, unrun. No real-file or restoration pair is accepted yet.
Standing owner approval for autonomous completion covers this concrete free
diagnostic packet and temporary CI. Current source is
`c554e86f38c1a64cbb73dd02ff99bf9129f6c76c`, tree
`5c83df17d08d888d4bd9cfd5c3ac5d51ed9304d0` (31 blobs / 242,315 bytes).
The [80 authored cases](../results/2026/11-02/fixture-checks.md) remain closed.
No application implementation or supported-format contract is changed.

The [earlier selection research](../research/2026/11-07/real-artifact-selection.md)
did not verify an F16/Q4_0 restoration pair. Its 50 MiB aim was our preference,
not an owner/platform limit. Complete public-file observations can resolve
actual encodings without inference, conversion, a paid plan or a local install.
The [product](../spec.md) and [format](../format.md) keep byte/container results
separate from recovered model behavior.

## Three frozen fixture identities

Read only these anonymous public HTTPS assets, once each; do not substitute
mutable main revisions or files with similar names.

| ID | Immutable publisher path | Bytes | SHA256 |
| --- | --- | ---: | --- |
| stories260k-f32 | [ggml-org/test-model-stories260K / stories260K-f32.gguf](https://huggingface.co/ggml-org/test-model-stories260K/blob/479896ec924af6d40fd419ab8f4d1eb2101de00d/stories260K-f32.gguf) | 1,185,376 | `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d` |
| stories15m-reference | [ggml-org/models-moved / tinyllamas/stories15M.gguf](https://huggingface.co/ggml-org/models-moved/blob/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M.gguf) | 98,357,920 | `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a` |
| stories15m-q4_0 | [same revision / stories15M-q4_0.gguf](https://huggingface.co/ggml-org/models-moved/blob/4cd12190a40543c61bad59a1b8d982dfc48de020/tinyllamas/stories15M-q4_0.gguf) | 19,077,344 | `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04` |

Total 118,620,640 bytes. The download URL replaces `blob` with `resolve` at the
same exact publisher/revision/path. Full byte count and SHA256, not labels or
HTTP success alone, bind identity. The official small-file rename identifies it
as F32. The untyped 15M reference is not thereby F16. The model card's MIT
declaration is publisher provenance, not a grant from Quantopsy or proof of
cryptographic conversion lineage. No downloaded artifact is redistributed.

An [upstream report](https://github.com/ggml-org/llama.cpp/issues/7711) describes
one Q8_0 tensor in the named 15M Q4_0 model. It is a compatibility warning,
not inventory evidence bound to these hashes. Current Quantopsy admits only
F32/F16/Q4_0 globally; inspect originals and independent descriptors must decide
actual admission. Never infer unknown-type payload lengths to make it pass.

## Actual observations and separate conclusions

Use Tier 2 pinned public-file fixtures, with no DB/production account. The
operational entry is `tests/scripts/run-public-gguf-observation.py`, supported
by `tests/helpers/public_gguf_inventory.py`; the inactive workflow example is
`public-gguf-observation.workflow.yml.example`. No app logic is mocked.

For each fixture:

1. Download into a newly owned runner directory with verified TLS, no auth or
   proxy inheritance, strict reviewed Hugging Face/CDN HTTPS redirect hosts and
   at most five redirects. Reject unknown hosts, non-HTTPS, credentials, excessive
   bytes, changed identities, non-identity encodings and nonterminal downloads.
   Keep signed redirect query data out of retained diagnostics.
2. Verify the complete size/hash before any product command. Independently read
   bounded GGUF header, typed metadata and all ordered tensor descriptors.
   Retain the 24-byte header and descriptor-region originals, offsets, lengths
   and hashes, plus per-metadata-entry encoding hashes/counts and allowlisted
   structural scalars. Do not retain tokenizer string arrays or weight payloads.
   Record numeric tensor types without guessing bytes for unknown encodings.
3. Run the real `python -B -S -m quantopsy inspect <owned-file>` once, in the
   frozen checkout, with a sanitized environment. Retain original stdout/stderr,
   exit/signal/reaping/EOF/group evidence. Validate a successful inspection's
   file identity and complete inventory against the independent observation.
   A clean finite CLI refusal remains `applicationAdmission.accepted: false`
   with its original error/exit; traceback, interruption, incomplete streams or
   other abnormal conditions fail the observation packet.
4. Rehash inputs and source at documented boundaries. Observe exactly three
   complete inventories/CLI outcomes or retain a failed partial packet.

`packetStatus: observed` means only that these bounded, hash-bound observations
completed and their owned resources closed normally. It is not an assertion that
all containers or the pair were admitted. Keep per-file admission booleans and
actual refusals separate. Keep `pairRestorationAdmission.accepted` false unless
both files are actually admitted, complete tensor inventories and encoded
metadata satisfy the product pairing contract, unquantized payload hashes agree,
and explicitly named `blk.0.attn_q.weight` is F16 reference versus Q4_0 candidate.
Record actual failed gates, not a pre-authored positive pair or guessed lineage.

No diagnose/probe/plan/repair/conversion, model engine/tokenizer, generated task,
quality score or inference runs. No new eval suite or acceptance threshold is
invented. A false pair is a useful negative result and remains false even if
the diagnostic workflow itself completes successfully.

## Resources and publication boundary

One free public Ubuntu job on `qa/quantopsy-artifacts-20261107-1` is capped at
15 minutes. Use the runner's supported preinstalled Python 3.11+ standard library,
record its exact version/binary hash, and install nothing. Preserve original
HOME while keeping models/temp/config under a unique 0700 root with a 0600 nonce.
Bound logical task storage to 512 MiB, work to 600 seconds clamped by the
earliest job-step clock, reserve 90 seconds cleanup and 30 startup/output margin,
and retain bounded original streams/reports. Each download/child has a finite
subdeadline; no retries. Uncertain process ownership or closure stays failed.
Only positively attributed UID/PID/starttime/groups may be signalled.

Preparation may append scoped AGENTS instructions and add only this proposal,
suite README, independent observer, runner and inactive workflow: 36 blobs.
All 30 other baseline paths and every product/test input remain exact. The
temporary active workflow exists only on its QA branch, pinned to reviewed
preparation; disable/retire it after a terminal result. A concrete changed
repair is required before another invocation. Preserve every original failure.

No weights or tokenizer payloads enter Git, logs, caches outside ownership or
the result report. Verify owned-file identity/hash before removal; retain source,
results, unrelated work and all existing files. App/source integrity, downloaded
file identity, parsed metadata, CLI outcome and cleanup are distinct evidence.
No host-wide network/write/process or kernel-interruption guarantee is claimed.

Storage Guard failed at 11.6 GiB free versus 34 GiB required during this work.
Only small source/docs/AST/metadata work is permitted locally; there is no local
download/build/runtime exception. No account, credential, provider/model call,
production data, DB, package/dependency, deployment or paid service is involved.
