# Initial Quantopsy source delivery

29 October 2026. Source is published in the separate public personal
[Quantopsy repository](https://github.com/alvarovillalbaa/quantopsy) at
[`685ad5f9b72f47f5b5fc3a609419e76ff31c56d2`](https://github.com/alvarovillalbaa/quantopsy/commit/685ad5f9b72f47f5b5fc3a609419e76ff31c56d2).
The full project remains open at its real-model inference and behavior
acceptance gates.

The complete one-line [Notion brief](https://app.notion.com/p/3fda1ee8bd0680eda403c2d43ba08224)
was read. Bounded local/project-owner searches found no existing Quantopsy
checkout or matching personal repository. A separate source repository was
created without borrowing model weights, environments or credentials.

The source implements a bounded GGUF reader and F32/F16/Q4_0 decoders, explicit
paired parameter diagnostics, supplied linear-response probes, and a bound
create-only selected F16 restoration path for dense Llama Q4_0 matrices. It
rebuilds and verifies a new candidate container, preserving unselected payloads
and installing its completion manifest last. Neither input is rewritten.

Independent source reviews found and corrected: an unnecessary zero-padding
constraint; architecture metadata validation; unbounded decoded metadata item
totals; numerical norm overflow/underflow; CLI argument/empty-output handling;
JSON FIFO blocking and descriptor cleanup; and output-reader inode binding.
These are source-review corrections, not observed passing runtime checks.

The [80-case proposal](../../../proposals/fixture-checks.md) and
[static inventory](source-inventory.json) are prepared without importing or
executing the product/tests. Static verification records only syntax, relative
imports, JSON/Markdown navigation, source review, bounded file inventory and
whitespace checks. There is no fixture, decoder, CLI, model or inference runtime
result yet. Source targets Python 3.11+ on POSIX macOS/Linux; the observed
existing executable is Python 3.14.7, not a verified compatibility matrix.

The preparation seal contains 25 inputs totaling 185,312 bytes, including 15
Python modules parsed with the Python 3.11 syntax profile. All 80 literal case
names are unique; 117 import records and 16 local Markdown links were checked
statically. Example JSON and direct UTF-8/whitespace checks passed. Dated result
files, including the seal itself, are explicitly outside that source snapshot.

No dependency, CI, runtime provider, Vercel/Supabase project, account, paid plan
or model was provisioned. The local storage preflight returned insufficient
reserve for large work, so no install/build/checkpoint copy was attempted.
This small source-only implementation does not bypass that limit.

Connected GitHub ref/tree readback matched all 27 file paths, modes, types and
blob SHAs, without truncation, against the initial source commit. The checkout
was clean. Direct leaf comparison was used instead of the connector's root
tree SHA. An authenticated workflow-definition read returned zero; the
connected Fetch endpoint does not expose that list. The
[machine publication receipt](source-publication.json) preserves these facts.
No CI file was created or job activated. Staged and unstaged diff checks passed
after removing one extra blank line at EOF; the source seal was refreshed
before committing, with no domain execution.

Approval was requested against the immutable published 80-case proposal and
remains pending. These dated publication records do not change any of the 25
sealed source/test/document inputs. No fixture run is implied by publication.

Next: obtain explicit approval for the published 80-case scope, execute and
retain those results, then prepare separate concrete real-model acceptance
once a licensed model pair, inference engine, capacity and frozen tasks are
available. Numerical/format success cannot substitute for behavior recovery.
