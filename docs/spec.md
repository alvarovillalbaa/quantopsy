# Quantopsy product contract

Owning [Notion brief](https://app.notion.com/p/3fda1ee8bd0680eda403c2d43ba08224):
“A tool that investigates and repairs behavior lost through model quantization”.
The complete returned brief was read on 29 October 2026; its unverified snapshot
and last edit are 20 October 2026, 17:51:33.117Z. It specifies no model, behavior
suite, inference engine, account, hosting provider or paid budget.

The full outcome requires credible diagnosis and recovery of lost model
behavior, with reference/quantized/candidate inference on frozen tasks and
regression limits. This first implementation supports actual local GGUF
inspection, paired numerical diagnostics, supplied linear activation probes and
selected weight restoration into a new candidate. It must not substitute weight
error or a successfully written container for that behavioral acceptance.

## Supported source behavior

Python 3.11+ standard library on POSIX macOS/Linux, no installation dependency
or network service. Windows is outside this descriptor-based implementation.
`inspect` reads a bounded GGUF container without loading model code. `diagnose`
compares explicitly selected tensors; `probe` applies paired rank-two weights
to supplied calibration vectors and reports output distortion. Neither metric
is a task-quality or behavioral-recovery score.

`plan` binds a compatible reference/quantized pair, explicit dense Llama tensor
selection, declared model/revision/license provenance, selected byte hashes and
the caller's added-weight budget. `repair` rechecks that plan and writes a NEW
bundle containing a GGUF candidate and a completion manifest. Inputs are never
rewritten. No automatic model download, tensor selection, promotion or inference.

Supported containers are single-file, little-endian GGUF v3 with only F32, F16
and Q4_0 tensors. Quantized rows must contain whole 32-value Q4_0 blocks. Other
types, shards, malformed/overlapping spans and excessive work fail visibly.
Repair initially targets rank-two dense `llama` matrices:
`blk.N.attn_{q,k,v,output}.weight` and `blk.N.ffn_{gate,up,down}.weight`.
Selected candidate tensors must be Q4_0 and reference tensors F16. This avoids
adding another reference rounding step. Expert, adapter, fused and multimodal
repair are outside the supported path, not silently interpreted as dense data.

Each input is at most 8 GiB; metadata at most 32 MiB, 8,192 tensors and 4,096
metadata entries. Decoded metadata has a separate 1,048,576-value total budget;
nested metadata arrays and zero-tensor tokenizer-only files are unsupported.
Inputs must be regular files, not symlinks or streams. The default selected work
budget is 64 million elements and the output limit 16 GiB. Streaming uses
bounded chunks. These are product limits, separate from the operator's storage
preflight/reserve policy.

Pairing compares architecture, metadata except an explicit quantization-only
allowlist, tensor names/shapes, and identical unquantized payloads. Declared
revision/license provenance and complete input hashes are mandatory for a plan.
These establish compatibility and supplied identity, not cryptographic lineage
or permission to redistribute third-party weights.

Preserve every unselected payload byte. Copy selected F16 payload bytes exactly,
rebuild aligned offsets and verify the candidate by reparsing/hash comparison.
Omit optional `general.file_type`, whose majority-encoding description may be
misleading after repair; record that change. Preserve the candidate's supplied
`general.quantization_version` when quantized tensors remain.

Create-only output rejects existing paths/symlinks. A manifest installed last is
the completion marker. Multiple files are not an atomic transaction; interruption
or unexpected output changes must leave a visible failure and remove only owned
temporary/partial files when ownership is still established. Never remove a
pre-existing directory to make a repair succeed.

## Acceptance and remaining scope

Tiny, independently encoded GGUF fixtures must verify the codec, numerical
diagnostics, pairing, real tensor replacement, unchanged payloads, rebuilt
offsets, metadata, budgets, tampering, overwrite refusal and failure cleanup.
These domain checks need the supplied AGENTS.md §7 approval before execution.
No fixture is an actual LLM or evidence of recovered behavior.

Real-model acceptance still needs a selected licensed model pair, storage and
compute capacity, an approved local inference engine, frozen tasks, paired
reference/quantized/candidate results and regression thresholds. Broader GGUF
types/formats and distribution remain open. This source slice does not complete
Quantopsy or the portfolio objective. No Vercel/Supabase backend is needed for
the current local CLI; no hosted replacement is claimed.
