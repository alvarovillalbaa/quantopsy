# First real-artifact selection — no accepted pair

7 November 2026. Bounded primary-source research covered the official
`ggml-org/test-model-stories260K`, `ggml-org/tiny-llamas`,
`ggml-org/models-moved/tinyllamas` and upstream `karpathy/tinyllamas` leads,
not every publisher. No compatible ready-made F16/Q4_0 pair was verified.

The [owning profile](../../../format.md) requires every tensor encoding to be
F32/F16/Q4_0, complete compatible inventories/metadata, and selected dense Llama
matrices to be Q4_0 versus F16. The [80-pass authored-fixture receipt](../../../results/2026/11-02/fixture-checks.md)
remains separate evidence; it used no trained checkpoint or inference.

## Verified official identities and incompatibilities

- [`stories260K-f32.gguf`](https://huggingface.co/ggml-org/test-model-stories260K/blob/479896ec924af6d40fd419ab8f4d1eb2101de00d/stories260K-f32.gguf),
  revision `479896ec924af6d40fd419ab8f4d1eb2101de00d`: **1,185,376 bytes**,
  SHA-256 `270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`.
  The [official rename](https://huggingface.co/ggml-org/test-model-stories260K/commit/479896ec924af6d40fd419ab8f4d1eb2101de00d)
  corrects the misleading former `stories260K-f16.gguf` name to **F32** without
  changing its contents. It is unsuitable as the required F16 restoration
  reference. The inspected official trees list no matching 260K Q4_0 sibling.
- The [immutable tiny-llamas upload](https://huggingface.co/ggml-org/tiny-llamas/commit/6e091d820cbe8f22eeb604d136403eca290b8c1e),
  revision `6e091d820cbe8f22eeb604d136403eca290b8c1e`, records:

  | File | LFS bytes | SHA-256 |
  | --- | ---: | --- |
  | `stories15M.gguf` | 98,357,920 | `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a` |
  | `stories15M-q4_0.gguf` | 19,077,344 | `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04` |

  Total: **117,435,264 bytes, approximately 112 MiB**. The 50 MiB pair aim is
  our conservative first-check preference, not a user/platform limit or proof
  that 112 MiB cannot run. F16 selection and full compatibility remain unverified.

## Provenance and possible preparation

The [upstream model card](https://huggingface.co/karpathy/tinyllamas/raw/main/README.md)
declares MIT and Llama 2 architecture trained on TinyStories. Its
[Stories260K training record](https://huggingface.co/karpathy/tinyllamas/commit/cbffbc90324256436c13dc6b4592c82c1e716354)
at revision `cbffbc90324256436c13dc6b4592c82c1e716354` specifies dimension 64.
Therefore `blk.0.attn_q.weight` is a **source-derived expectation** of 64×64 =
4,096 elements, **not verified GGUF inventory**. This declaration does not prove
the converted files' cryptographic lineage.

A pinned conversion of that trained source into F16 and Q4_0 is possible future
preparation, not implemented or accepted. All-tensor admission, metadata equality,
unchanged payloads and complete hashes would still require verification.
No weight payload was retrieved or saved; no inference, CLI/model execution,
tests, installation, CI or account action occurred. No pair or recovered behavior
is accepted by this research.
