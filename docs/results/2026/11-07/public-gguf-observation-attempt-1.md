# Public GGUF observation — first control remained UNRUN

**UNRUN registration gap; no runtime failure or admission result.**
Polling stopped on Root's instruction after three exact-head reads returned no
runs, Actions were confirmed enabled, and the owned workflow file was verified.
No workflow/run/job handle or terminal log was observed. No emitted result JSON
or job-log file is fabricated.

- Repository: `alvarovillalbaa/quantopsy`.
- QA branch: `qa/quantopsy-artifacts-20261107-1`.
- Root-created activation at 12:21:04 UTC on 7 November 2026:
  `e2bee5fbc149eae22f422358a12b7e77f1df494a`;
  activation tree `df37a0f137701d43ed8dc5f4808d632df8c606eb`.
- Prepared source: `c574dbd713a54efc744632235e92f49114aa1a96`,
  direct parent `c554e86f38c1a64cbb73dd02ff99bf9129f6c76c`;
  prepared tree `01c933c2aa14971861629d1685b7490b3cfd5a56`
  (Root-sealed 36 blobs / 346,060 bytes).
- Owned active path: `.github/workflows/public-gguf-observation-1.yml`,
  1,186 bytes, Git blob `b628b4d3340349c159a6b82919d2e36e2f9fa61e`.

The three discovery reads used generic `github_fetch` on unfiltered REST
`/repos/alvarovillalbaa/quantopsy/actions/runs?head_sha=e2bee5fbc149eae22f422358a12b7e77f1df494a&per_page=10`.
They did not use the pull-request-only commit-workflow wrapper. Each returned
`total_count: 0` and `workflow_runs: []`:

| UTC observation | Exact-head workflow runs |
| --- | ---: |
| 12:21:24 | 0 |
| 12:22:26 | 0 |
| 12:23:50 | 0 |

At Root's request, one direct `gh api --method GET` cross-check at 12:25:55 UTC
used `/repos/alvarovillalbaa/quantopsy/actions/runs` with parameters
`head_sha=e2bee5fbc149eae22f422358a12b7e77f1df494a`, `event=create`, `per_page=10`.
It independently returned `total_count: 0` and `workflow_runs: []`. This was a
read-only discovery-authority check, not a dispatch or runtime retry.

Separate supported GitHub REST reads, bracketed between 12:22:26 and
12:23:24 UTC, returned repository Actions `enabled: true`,
`allowed_actions: all` and workflow list `total_count: 0`.
The exact activation's contents API confirmed the owned active path, size and
blob above by 12:23:24 UTC. These are read observations, not a diagnosed reason
why registration/event execution was absent.

The frozen runner, independent helper and workflow source remain unchanged:
runner SHA256 `4c98914bcc3d0aef51b4817f5d25fa17ac716e17ed2c3341df5ea3be7d66d9da`,
helper `e1a51e561fc9ba2ab0a2b4e3604575accd37d2b560eebd7cda62772bda32c1f5`,
inactive example `4e314350e048bd1f8fe4c24dffd16da96be807398bdc83f0c3dfa4c1be3626e5`.

No artifact download, inventory helper, Quantopsy CLI, pair assessment or
teardown execution was observed. All three per-file admissions and restoration
pair admission remain unmeasured; diagnostic completion and product acceptance
are not claimed. The closed 80-case fixture result is unchanged and separate.

No dispatch, retry, branch deletion, default-branch bootstrap, source change or
unknown-workflow disable was performed. Root coordinates any concrete changed
control scope. There are no weights, tokenizer payloads or runtime outputs in
this note. No local download, installation, model execution or CI launch occurred
from the monitoring agent.
