# Quantopsy check layout and data

The existing `tests/test_*.py`, `tests/fixtures.py` and 80 authored cases remain
grandfathered in place. Their [recorded fixture run](../docs/results/2026/11-02/fixture-checks.md)
is closed and is not repeated by the public-artifact packet.

| Deliberate entry | Layer / tier | Data and claim |
| --- | --- | --- |
| `scripts/run-public-gguf-observation.py` | Operational CLI/file integration, Tier 2 | Three immutable publicly published GGUF files used as pinned fixtures; actual parser/CLI observations, without inference or a database |
| `helpers/public_gguf_inventory.py` | Support for that entry, same Tier 2 | Independent read-only structural observer; no product helper imports, app logic mocks or weight execution |

These are real public trained-container bytes used as fixture inputs, not
independently authored mock weights. Tier 2 describes the isolated static fixture
boundary; it does not turn their provenance into synthetic data or imply a
database replica. Tiers 3–5 and production-role access are not used. No collected
unit test is added and no `tests/evals/` or model-quality gate is invented.

The [scoped contract](../docs/proposals/public-gguf-observation.md) fixes the
source, three complete input identities, runner ownership/bounds and teardown.
The report must say `packetStatus: observed` only for complete hash-bound,
normally closed observations. Each `applicationAdmission.accepted` and
`pairRestorationAdmission.accepted` is separately factual; refused/nonzero CLI
outputs stay preserved and false. A completed observation packet is not a
passing restoration/behavior test. Unexpected/partial/network/hash/ownership
failures remain failed. No inference, selection automation, repair, provider,
private corpus, DB, account, dependency or paid service belongs to this packet.

Candidate 1's create event produced no observed workflow run; its [UNRUN note](../docs/results/2026/11-07/public-gguf-observation-attempt-1.md)
is preserved, without fabricated runtime outputs. Candidate 2 changes only
event/ref/source guards and dispatch controls. Its default-branch registration
stub stays inert; a fixed-source job may execute only on the exact reviewed QA2
ref. Domain logic, public inputs and acceptance distinctions remain unchanged.

The [actual second-control receipt](../docs/results/2026/11-07/public-gguf-observation-attempt-2.md) observed all three fixed
files on Linux/Python 3.12.3: inspect exits 0/0/2. This is diagnostic completion;
the Q8_0-containing file and F32-reference restoration pair remain unadmitted.
The first control is UNRUN, not a failed or passing runtime test. Raw originals,
source guards and weight teardown are preserved separately from future scopes.

## Separate opt-in v2 verification

The new `integration/test_profile_v2.py` suite uses Tier 2 independently authored
GGUF byte fixtures, capped at 64 KiB by the existing builder. Its helper belongs
under `helpers/`; no product encoder constructs expected bytes. The suite covers
the file/CLI/profile/rebuild seams without a database, network or model engine.
Unittest module/class names declare the integration/Tier 2 scope; the older
flat test modules remain grandfathered. Existing reader/repair/CLI modules total
64 cases and are relevant regressions after these source changes. The unchanged
16 diagnostics cases are excluded from this packet.

A separately reviewed `scripts/` controller ran those cases once in owned
free public CI, then used only the exact pinned 15M reference/quantized public
files for v2 inspect, selected F32/Q4 plan, repair and candidate inspection.
Every unquantized binding must be freshly evaluated, including the fixture's
13 F32 payloads; every unselected output payload, including Q8, must retain its
input hash. Source/input identity, budgets, originals, manifest-last completion,
process closure and temporary teardown are required. The raw model/candidate
bytes stay in owned disposable runner storage, never Git or local storage.
This is Tier 2 isolated static public-fixture integration, with actual container
bytes rather than synthetic weights. No database tiers or production roles.

The [first v2 receipt](../docs/results/2026/11-07/gguf-compat-v2-attempt-1.md)
records 92 passes (28 new plus 64 regressions), both container admissions,
13 fresh unquantized bindings and all 57 prescribed candidate payload hashes.
Source `e18dee05` ran on preinstalled Linux/Python 3.12.3; 41 commands closed
normally with zero exits and verified owned teardown. The 16 diagnostics cases
were not repeated. Historical v1 results remain unchanged. No model inference,
recovered behavior or full-product acceptance follows from this byte-level run.

## Frozen behavioral slice

`test_behavior.py` adds 35 authored Tier 2 data-only comparison cases with
`behavior_fixtures.py`: exact Decimal decisions, nonvacuous recovery, regressions,
coverage/order, strict identities/manifests, same-read hashes and create-only CLI
outputs. These use fake fingerprints and measurements, never actual inference.
The new packet runs them with the 16 relevant existing CLI cases (51 total).

The [separate approved packet](../docs/proposals/behavioral-inference-v1.md)
then performs actual teacher-forced CPU forwards on two pinned public models
and one manually predeclared 42-matrix candidate. Trusted helper/scorer/build
source lives under `helpers/`; the controller under `scripts/`. The product does
not load an engine. Original token scores, frozen identities, source/process
guards and owned teardown are required independently of comparison records.
Experiment completion and the frozen recovery verdict remain separate. This
small story suite cannot establish general or semantic behavioral recovery.

The [first invocation](../docs/results/2026/11-08/behavioral-inference-attempt-1.md)
passed these 51 cases at `f2b3c812`; a later controller-only repair reuses that
closed result only after verifying its receipt and unchanged app/test bytes.
Actual inference and verified scratch removal did not occur in the first run.
