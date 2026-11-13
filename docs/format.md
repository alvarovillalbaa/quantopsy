# Container, diagnostics and repair contract

The [product contract](spec.md) keeps model behavior acceptance separate from
the byte-level and numerical work implemented here. The supported profile is
deliberately smaller than the [pinned GGUF specification](sources.md).

## Admission and file ownership

Inputs are unchanged regular files with no symlink leaf. The POSIX reader keeps
a read-only descriptor, checks the current pathname and held descriptor against
device/inode, mode, size and nanosecond timestamps, and streams with `pread`.
Full file hashes are freshly computed, not cached. Parsed tensor records belong
to their own reader. JSON inputs use bounded regular-file reads and reject
duplicate keys, nonfinite numbers, invalid Unicode and excessive nesting.

These checks detect source changes at the documented boundaries. They do not
provide a filesystem snapshot or coordinate with arbitrary concurrent writers.
Keep inputs and the output parent unchanged for the operation.

| Profile limit | Default |
| --- | ---: |
| Each input / rebuilt output | 8 GiB / 16 GiB |
| Encoded metadata / decoded metadata values | 32 MiB / 1,048,576 |
| Metadata entries / items in one array | 4,096 / 262,144 |
| Tensors / dimensions per tensor | 8,192 / 4 |
| String / metadata key / tensor name bytes | 1 MiB / 1,024 / 63 |
| Stream chunk | 1 MiB |
| Selected numerical work | 64,000,000 elements |
| Data-only JSON | 2 MiB, depth 16, 100,000 visited values |

Only GGUF v3 little-endian, F32, F16 and Q4_0 tensor types are admitted. Metadata
scalar/string arrays are supported; nested arrays, sharded metadata and
zero-tensor files are rejected. Tensor dimensions are positive. Payload spans
must be aligned, nonoverlapping and within the file. Gaps, header padding and
trailing bytes are opaque; the reader does not require their bytes to be zero.
Alignment defaults to 32, may be a positive multiple of eight up to 4,096, and
does not have to be a power of two. The 63-byte tensor name cap is this
implementation's interoperability profile, not the specification's full limit.

Q4_0 requires quantization version 2 and complete blocks along the first tensor
dimension. Each 32-value block has a binary16 scale and 16 packed bytes: the
low nibbles produce the first 16 values, then the high nibbles produce the next
16; each signed level is the nibble minus eight, multiplied by the scale.
Finite negative and zero scales are valid. Unknown encodings are rejected
without guessing their byte sizes. Selected decoded values must be finite.

## Numerical evidence

`diagnose` explicitly selects one to 256 distinct tensor names. It reports
changed-value counts, root mean squared error, reference RMS, maximum absolute
error and relative L2 error. Relative L2 is unavailable only when the reference
norm is zero. Arithmetic that cannot produce a finite result fails visibly.

`probe` accepts the [example document shape](../examples/probes.example.json):
version, kind, tensor, provenance and vectors, with no extra fields. Provenance
requires nonblank model ID, revision, dataset and capture labels. There are one
to 16 finite vectors, each at most 16,384 columns and exactly the selected
rank-two tensor's first dimension. The second dimension is the output row
count. Reference and quantized dot products use the same supplied vectors in
GGML row order; reports compare those outputs. The work bound includes the
number of vectors. The canonical input document hash is recorded.

These are parameter and supplied linear-response diagnostics. They do not
capture activation vectors, run a tokenizer/model, estimate task accuracy,
automatically select repairs or establish recovered behavior.

## Pairing, plan and candidate bundle

Pairing requires an F32/F16 reference, the same complete tensor inventory and
dimensions, and equal encoded metadata outside four quantization-only keys:
`general.file_type`, `general.quantization_version`, `general.quantized_by`,
`general.quantized_at`. Unquantized tensors retain their reference type and
payload hash. This binds compatibility; it does not prove model lineage.

Restoration supports dense `llama` attention q/k/v/output and FFN gate/up/down
rank-two matrices. Each selected quantized tensor is Q4_0 and its reference is
F16. Expert, fused, adapter and multimodal layouts are rejected. A version-one
plan requires [five provenance fields](../examples/provenance.example.json),
each nonblank and at most 2,048 UTF-8 bytes; full file hashes and sizes; selected
tensor hashes, shapes, types and byte counts; and explicit added-weight and
decode-work budgets. Extra fields, boolean integers and inconsistent arithmetic
are rejected. Plan validation returns detached records. License declarations
come from the operator and grant no redistribution rights.

Repair regenerates the plan against the currently opened inputs. It creates a
new directory, streams selected F16 bytes and unselected original payloads,
rebuilds aligned offsets, and retains the candidate metadata order/bytes except
optional `general.file_type`. That majority-encoding label can become misleading
after restoration; omission is recorded. The new container need not preserve
opaque input gaps or trailing bytes. Added-weight bytes measure payload growth,
not an exact change in whole container size.

The candidate is reparsed, checked against the owned output inode, and verified
against every prescribed tensor payload and complete input hash. The bundle is:

- `plan.json`: the accepted canonical repair plan.
- `candidate.gguf`: the verified candidate container.
- `repair.json`: installed last as the completion marker, binding the plan,
  output/input hashes, provenance, selected tensors, budgets and verification.

Files are mode 0600 inside a mode 0700 new directory. Existing output paths are
refused, including empty directories and symlinks. Temporary files are installed
with create-only hard links. The manifest still says
`behavioral_validation: "not_run"`. A multi-file bundle is not an atomic
transaction; interrupted or externally changed output must not be treated as
complete. Failure cleanup removes only tracked owned inodes and an empty owned
directory; unexpected/replaced files are preserved. If ownership or cleanup
cannot be established, inspect any residue before retrying in a new directory.

The default writer reserve is 1 GiB after estimated output. This does not
override a stricter local storage policy; perform that preflight before large
checkpoints or outputs. No credentials, model weights or local probe artifacts
belong in this repository.

## Opt-in profile v2

The preceding v1 contract and defaults remain unchanged. `inspect`, `plan` and
`repair` accept an explicit `--profile v2`; omission means v1. The Python reader
and repair APIs likewise require matching explicit profiles on both inputs and
the operation. A plan's version never chooses the operation profile. V2
inspection, plans and completion manifests use integer `schema_version: 2` with
their existing kind identifiers and strict field shapes; cross-version records,
boolean versions and unknown profiles are refused. `diagnose` and `probe` have
no v2 path.

V2 admits Q8_0 (type 8) only as opaque payloads. Its pinned block layout is 34
bytes per 32 values; dimension zero must contain complete blocks. Span bounds,
alignment, overlap and file-size rules still apply. This profile conservatively
requires integer quantization version 2 whenever Q4_0 or Q8_0 occurs, including
Q8-only files; that is a Quantopsy policy, not a general GGUF format mandate.
`iter_values` explicitly refuses Q8 numerical decoding.

Selected v2 restoration requires F32 reference payloads and rank-two Q4_0
dense-Llama targets; v1 retains its F16 reference requirement. The reference
container still contains only F32/F16 tensors. Complete metadata/name/dimension
agreement and type/hash equality for every unquantized input tensor remain
mandatory. The observed public fixture has 13 such bindings; the implementation
uses the complete actual inventory rather than a hard-coded count.

Selected F32 and Q4 values must be finite within the same explicit work and
growth budgets. Every unselected payload, including Q8, is copied unchanged.
Both partial and installed candidate readers use v2; inventory, metadata,
selected/unselected hashes and complete input hashes are verified before the
last completion manifest. No Q8 restoration, automatic selection, model engine,
inference, cryptographic lineage or behavioral recovery is implied. See the
[separate proposal](proposals/opaque-q8-f32-v2.md) for its verification gate.

## Opt-in profile v3

V3 retains v2's container types, F32-to-Q4_0 selected restoration and opaque Q8_0
copying, adding only explicit `token_embd.weight` selection for dense Llama.
Inspection, plans and completion manifests use integer version 3; both readers
and the operation must explicitly agree. Older plans and profiles are not
upgraded implicitly. `diagnose`/`probe` and Q8 numerical admission are unchanged.

The selected rank-two input embedding must have dimensions equal to positive
uint32 `llama.embedding_length` and the count of a nonempty string-array
`tokenizer.ggml.tokens`. Optional uint32 `llama.vocab_size` must agree with that
count. Paired encoded metadata and tensor shapes must already match. This
conservative shape policy precedes selected-value decoding and actual writing.
F32/Q4_0 finite-value, work/growth, full input and payload hashes, create-only
ownership, unselected bytes and manifest-last verification remain required.
The writer regenerates the complete plan from the actual opened inputs.

V1/v2 still refuse this target. V3 does not admit `output.weight`, selected Q8,
automatic selection or tied-weight inference. The [contract](proposals/input-embedding-v3.md)
and [fixture receipt](results/2026/11-13/profile-v3-checks.md) state verification
limits; the separately frozen inference study remains unrun.
