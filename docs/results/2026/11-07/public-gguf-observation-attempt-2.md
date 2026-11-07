# Public GGUF observation — attempt 2

**Diagnostic complete; restoration-pair admission false; product acceptance false.**
The [first control](public-gguf-observation-attempt-1.md) remains UNRUN.
This records actual public-container observations, not new fixture-test passes.

- [Run 42373655591](https://github.com/alvarovillalbaa/quantopsy/actions/runs/42373655591),
  job `128406740804`, workflow `410321779`: terminal success.
- Prepared source `a604b0df2c176630ddcc8874477dea7e797ce01f`;
  activation `0c4e4a07071adfd8e85b4650395e879ebfd52097`, exact QA2 dispatch.
- Run created 7 November 2026 at 12:42:15 UTC; observer ran
  12:42:24.745–12:42:29.488 UTC. Preinstalled Python 3.12.3; no installation.
- Owned workflow disabled after terminal; fresh REST read at 12:43:34 UTC
  confirmed `disabled_manually` (updated 12:43:33 UTC).

The single full job-log read was saved before parsing. Its unique report
payload reproduces the JSON file, including its final newline.

| Original | Bytes | SHA256 |
| --- | ---: | --- |
| [JSON](public-gguf-observation-attempt-2.json) | 215,966 | `f5cd4be7e15f884eb5e1527f711ab0c83259e855d273825651dbd75d6a96ee54` |
| [Job log](public-gguf-observation-attempt-2.log) | 233,506 | `9bdfb787b77119564f6e5a9e9be2570bdd242d1e6964877bc39d17c36e77bf72` |

All three immutable inputs completed HTTP 200 after one observed HTTPS redirect
each to `us.aws.cdn.hf.co`; signed queries were redacted. Content lengths,
received/written bytes, full SHA256 and file identities matched the fixed pins.
Total downloaded bytes: 118,620,640. No weights are retained in these results.

| File | Bytes | Metadata / tensor inventory | Actual inspect |
| --- | ---: | --- | --- |
| stories260K-f32.gguf | 1,185,376 | 19 / 48 F32 | exit 0, admitted |
| stories15M.gguf | 98,357,920 | 18 / 57 F32 | exit 0, admitted |
| stories15M-q4_0.gguf | 19,077,344 | 20 / 43 Q4_0 + 13 F32 + 1 Q8_0 | exit 2, refused |

Their observed full SHA256 values, respectively:

- `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`
- `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`
- `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04`

The Q8_0 descriptor is `output.weight`, dimensions `[288, 32000]`.
The original 136-byte error JSON reports `gguf_tensor_type_unsupported`;
no Q8_0 payload length was guessed. The explicit selected
`blk.0.attn_q.weight` is `[288, 288]`: reference F32, quantized Q4_0.
The required F16/Q4_0 selection is absent. The work gate is therefore
unestablished, not an observed size overrun. Complete names/dimensions and
encoded metadata outside the four quantization keys match, but global encoding
and application admission fail; unquantized payload bindings were not evaluated.
No repair plan, numerical validation, cryptographic lineage or behavior is proved.

Independent audits decoded all 72 header bytes and 162 ordered descriptors,
matched actual CLI outputs/refusal, and verified all 81 retained gzip records:
78 original command streams plus three descriptor-region copies, 137,466 decoded
and 44,279 compressed bytes. Copies are evidence duplication, not extra commands.
All 38 source bodies (355,067 bytes), Git blobs/modes, ten original tree/index
inventories and five guards match prepared source; vector
`e330dbd7d0c3bb67b4374496942b7c7693e8c6e034d6569cfd8d1652e693da92`.
All 32 unmodified baseline blobs and all 33 prior modes match; AGENTS extends its
baseline prefix. The inert stub and original UNRUN note are unchanged.

These source statements refer to the immutable preparation; later status docs
and stub retirement are separate. Raw GGUF metadata-region bytes and weight
payloads were not retained or locally rehashed after cleanup. Recorded hashes
and allowlisted structural values remain. Their full-file/encoded hashes are
source-bound CI observations, not an independent local artifact replay.

All 39 recorded command children reached EOF, were reaped and had absent
original process groups: 38 exit 0, one retained inspect exit 2.
`allCommandExitsZero` is false. No supervisory signals or uncertain/remaining
owned processes; all five owned files were hash/identity checked before removal.
Peak sampled scratch was 118,643,196 bytes; owned scratch removal passed.
The 15-minute job / 600-second work / 90-second cleanup / 30-second margin
limits remained in force. No host-wide process or network-absence claim is made.

No inference, conversion, repair, tokenizer/model-code execution, provider use,
closed 80-case rerun, automatic substitution or retry occurred. A supported
matched F16/Q4_0 restoration pair and recovered behavior remain unverified.
