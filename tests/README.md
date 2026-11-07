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
