# Owning and primary sources

Read on 29 October 2026. No model code, weights or third-party library was
vendored. Format rules inform an original standard-library implementation;
the resulting source has not yet passed its proposed fixture checks.

- [Quantopsy Notion brief](https://app.notion.com/p/3fda1ee8bd0680eda403c2d43ba08224)
  owns product intent. The complete returned brief calls for investigation and
  repair of behavior lost through quantization. Its last-edit snapshot is
  20 October 2026, not a promise of subsequent live changes.
- [GGUF specification at ggml facdc44](https://github.com/ggml-org/ggml/blob/facdc44a67da17df4433adcc468b424f48e9ded2/docs/gguf.md)
  defines typed metadata, tensor descriptors, relative offsets and alignment.
  Quantopsy implements a bounded little-endian subset. Source review corrected
  an initial unnecessary zero-padding restriction before publication.
- [llama.cpp block definition at eb4e244](https://github.com/ggml-org/llama.cpp/blob/eb4e244be77a906a183cd4a00fd51d9ad3200955/ggml/src/ggml-common.h#L194-L200)
  and [Q4_0 decoder](https://github.com/ggml-org/llama.cpp/blob/eb4e244be77a906a183cd4a00fd51d9ad3200955/ggml/src/ggml-quants.c#L464)
  define the scale/packed-nibble layout used for the admitted encoding.
  [Quantization version](https://github.com/ggml-org/llama.cpp/blob/eb4e244be77a906a183cd4a00fd51d9ad3200955/ggml/include/ggml.h#L218-L223)
  and [dense/expert Llama source](https://github.com/ggml-org/llama.cpp/blob/eb4e244be77a906a183cd4a00fd51d9ad3200955/src/models/llama.cpp#L36-L105)
  informed the explicit version and restoration profile. Those pinned excerpts
  were reviewed by the delegated reader/writer author; no engine was installed.
- [AWQ paper, arXiv:2306.00978v6](https://arxiv.org/abs/2306.00978v6)
  motivates considering activation distributions when investigating quantization
  loss. Quantopsy's supplied linear probes are a separate, limited diagnostic;
  this source does not implement AWQ or reproduce its quality/performance claims.

The [format contract](format.md) documents product limits and evidence fields.
Actual model behavior remains governed by the [acceptance contract](spec.md).
