# BLAZE-X — Model Packager

A stable archive format for large language models with binary diff/patch, integrity verification, on-the-fly quantisation, and lossless export to standard formats.

---

## The Problem

LLM distribution has no standard packaging layer. A 70B model is a directory of `.safetensors` shards that you download in full every time, regardless of how much changed. Fine-tunes of the same base model are distributed as complete copies. There is no standard way to ask "what changed between these two models?" or to distribute only what changed.

BLAZE-X solves packaging and distribution.

---

## Benchmark Results
 
Tested on real base → instruct model pairs across two architectures. Patches are
bit-perfect — SHA-256 verified against the original target after apply.
 
| Model pair | Architecture | Full model | Patch size | vs full model |
|---|---|---|---|---|
| Qwen2.5-7B → 7B-Instruct | Qwen | 15.3 GB | 6.1 GB | **40.1%** |
| Qwen2.5-14B → 14B-Instruct | Qwen | 29.5 GB | 11.3 GB | **38.3%** |
| Llama 3.1-8B → 8B-Instruct | Llama | 15.0 GB | 7.7 GB | **47.9%** |
 
Distributing a model update costs **38–48% of a full re-download**, with zero
quality loss. The patch is applied locally against the base model — no rounding,
no approximation, the reconstructed model is identical to the original target
byte-for-byte.
 
Compression improves with model size: the 14B patch is proportionally smaller
than the 7B, consistent with larger models having a higher fraction of unchanged
weights after instruction tuning.
 
Results are architecture-agnostic. Qwen and Llama use different attention
implementations, tokenizers, and training pipelines — the delta codec operates
on raw BF16 weight bytes and makes no architecture-specific assumptions.
 
### Verification
 
```
✓ 339 tensors verified — SHA-256 OK   (Qwen2.5-7B-Instruct)
✓ 579 tensors verified — SHA-256 OK   (Qwen2.5-14B-Instruct)
✓ 291 tensors verified — SHA-256 OK   (Llama 3.1-8B-Instruct)
```
---

## What BLAZE-X Does

**Single-file archives** — pack a HuggingFace model directory into one `.blz` file. All safetensors shards, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.model` (SentencePiece binary), `generation_config.json`, `vocab.json`, and `merges.txt` are embedded automatically when present. The exported archive is a complete, self-contained drop-in replacement for the original directory.

**Binary diff and patch** — compare two archives tensor-by-tensor using xxh3 checksums. Changed tensors are encoded with **XOR + zstd** deltas (SplitStream for F16/BF16, sparse XOR for sparse integer tensors, full XOR otherwise); apply XOR-combines with the base tensor and verifies xxh3. Patches are typically far smaller than shipping full changed tensors. Older `.blzdiff` files (format v1) store raw tensor bytes and remain readable.

**Integrity verification** — every tensor stores an xxh3-64 checksum. The archive stores a SHA-256 over the entire data section. `blazex verify` checks both. A standalone Python script (`scripts/verify.py`) does the same without the binary.

**List and extract individual layers** — inspect what is in an archive and extract specific tensors without reading the whole file.

**Export with optional on-the-fly quantisation** — export back to SafeTensors, PyTorch-loadable raw binaries, or native GGUF v3 (directly loadable by llama.cpp and Ollama, no external tools required). Add `--cast` to convert weights during export: downsample to F16/BF16, or quantise to Q8_0, Q4_0, or Q4_K. The archive itself is untouched — casts only affect the exported output.

---

## Format

The `.blz` format is intentionally simple and stable:

```
[MAGIC 8B] [VERSION 4B] [HEADER_LEN 8B] [HEADER JSON] [RAW TENSOR DATA...]
```

- **Header** is JSON. Human-readable, debuggable with any text editor or `jq`.
- **Tensor data** is raw little-endian bytes in the original dtype. No reinterpretation.
- **Version field** is checked on open. Format changes will increment the version.
- **No compression in the archive itself.** Compress the `.blz` file externally with zstd or lz4 if you want smaller transfers. The format stays simple.

The patch container uses format version **2** (4-byte field after magic): a JSON manifest listing each tensor op (unchanged / modified / added / removed), plus a blob section. Each modified tensor’s blob is **`BLXD`-prefixed** compression (see `src/delta_patch.rs`). Format **v1** patches used raw tensor bytes only and are still supported on apply.

---

## Install

```bash
cargo install --path .
```

Or build the release binary directly:

```bash
cargo build --release
# binary at target/release/blazex
```

---

## Usage

### Pack a model

```bash
blazex pack --input ./llama3-8b-hf/ --output llama3-8b.blz
```

Reads all `.safetensors` shards and embeds every file needed to reconstitute the model directory on export:

| File | Purpose |
|------|---------|
| `config.json` | Model architecture config — always embedded |
| `tokenizer.json` | Fast tokenizer vocab & rules |
| `tokenizer_config.json` | Required by `AutoTokenizer.from_pretrained()` |
| `special_tokens_map.json` | BOS / EOS / PAD / UNK / MASK token definitions |
| `tokenizer.model` | SentencePiece binary (LLaMA, Mistral, T5, …) |
| `generation_config.json` | Default generation parameters |
| `vocab.json` | BPE vocabulary (GPT-2, RoBERTa, Falcon, …) |
| `merges.txt` | BPE merge rules (GPT-2, RoBERTa, Falcon, …) |

All files are stored verbatim — no JSON parsing, no reformatting. Binary files (`tokenizer.model`) are base64-encoded inside the header. All are written back byte-perfect on export.

### Inspect an archive

```bash
blazex info llama3-8b.blz
blazex list llama3-8b.blz
blazex list llama3-8b.blz --filter self_attn   # filter by name substring
```

### Extract specific tensors

```bash
# Extract everything
blazex extract --archive llama3-8b.blz --output ./tensors/

# Extract specific layers
blazex extract --archive llama3-8b.blz --output ./tensors/ \
  --tensor model.layers.0.self_attn.q_proj.weight \
  --tensor model.layers.0.self_attn.k_proj.weight
```

Each tensor is written as `<n>.bin` — raw little-endian bytes in the original dtype.

### Verify integrity

```bash
blazex verify llama3-8b.blz
```

Or without the binary (pure Python, no dependencies):

```bash
python3 scripts/verify.py llama3-8b.blz
python3 scripts/verify.py llama3-8b.blz --json   # machine-readable
```

### Diff two models

```bash
# Create a patch from base model to fine-tuned variant
blazex diff --base llama3-8b.blz --target llama3-8b-finetuned.blz --output delta.blzdiff
```

Output:
```
Diff summary:
  Unchanged : 218
  Modified  : 14
  Added     : 0
  Removed   : 0
  Patch size: 487.3 MB (5.2% of full model)
```

### Apply a patch

```bash
blazex apply --base llama3-8b.blz --patch delta.blzdiff --output llama3-8b-finetuned.blz
```

Every tensor's xxh3 checksum is verified during apply. If anything doesn't match, the operation aborts before writing output.

### Export

```bash
# Back to HuggingFace SafeTensors (lossless)
blazex export --archive llama3-8b.blz --output ./exported/ --to safetensors

# SafeTensors downsampled to F16
blazex export --archive llama3-8b.blz --output ./exported/ --to safetensors --cast f16

# PyTorch-loadable raw binaries + manifest.json + load_model.py helper
blazex export --archive llama3-8b.blz --output ./exported/ --to pytorch

# Native GGUF v3 — loads directly in llama.cpp and Ollama, no external tools
blazex export --archive llama3-8b.blz --output model --to gguf

# GGUF quantised to Q4_K
blazex export --archive llama3-8b.blz --output model --to gguf --cast q4_k

# GGUF quantised to Q8_0
blazex export --archive llama3-8b.blz --output model --to gguf --cast q8_0

# One .bin per tensor + index.json (universal interchange)
blazex export --archive llama3-8b.blz --output ./exported/ --to raw
```

### Cast targets

| Flag | Description | Size vs F32 |
|------|-------------|-------------|
| `f32` | No-op for F32 sources; upcasts lower precision | 1× |
| `f16` | IEEE 754 half precision | 0.5× |
| `bf16` | Brain float (same exponent range as F32) | 0.5× |
| `q8_0` | GGML Q8_0 — block of 32, one F16 scale, 32× i8 | ~0.25× |
| `q4_0` | GGML Q4_0 — block of 32, one F16 scale, 32× 4-bit | ~0.125× |
| `q4_k` | GGML Q4_K — 256-element super-block, sub-block scales | ~0.125× |

Q4_K generally gives better reconstruction quality than Q4_0 at the same bit rate. Q8_0 is nearly lossless in practice. Q4_K requires tensor element counts to be multiples of 256 — tensors that don't meet this (typically small norm layers) will error; use `--cast q8_0` or `--cast f16` for those.

All quantised formats exactly match the GGML on-disk layout, so GGUF files produced with `--cast` load cleanly in llama.cpp and Ollama.

---

## Python loader (PyTorch export)

After `blazex export --to pytorch`, load in Python:

```python
from load_model import load_tensors
tensors = load_tensors("./exported/")
# {'model.embed_tokens.weight': tensor([...]), ...}
```

Or manually:

```python
import json, numpy as np, torch
manifest = json.load(open("exported/manifest.json"))
tensors = {}
for t in manifest["tensors"]:
    raw = np.fromfile(t["file"], dtype=np.dtype(t["numpy_dtype"]))
    tensors[t["name"]] = torch.from_numpy(raw.reshape(t["shape"]))
```

---

## Patch verification tool

`blazex-verify-patch` is a standalone binary that performs an end-to-end correctness proof of the diff/patch pipeline against any existing `.blz` archive.

```bash
blazex-verify-patch model.blz
```

It runs six steps automatically:

1. Loads the archive and reports tensor count and data size
2. Builds a mutated "target" archive — corrupts a fraction of tensors, adds synthetic tensors, removes some
3. Creates a `.blzdiff` patch from base → target
4. Applies the patch to reconstruct the target from scratch
5. Compares every byte of every tensor in the reconstructed archive against the target — reports first differing byte and total mismatch count
6. Cross-checks archive-level SHA-256 and per-tensor xxh3 checksums independently on both archives

```bash
# Default: 15% of tensors corrupted, 2 added, 1 removed
blazex-verify-patch model.blz

# Full verbose output — shows ok/FAIL per tensor
blazex-verify-patch model.blz --verbose

# Scale mutation (multiply weights by 1.1 — simulates fine-tune drift)
blazex-verify-patch model.blz --mutation scale --mutate-fraction 0.3

# Zero mutation, keep temp files for inspection
blazex-verify-patch model.blz --mutation zero --keep-tmp

# Reproducible run with fixed seed
blazex-verify-patch model.blz --seed 12345
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--mutate-fraction <f>` | `0.15` | Fraction of tensors to mutate |
| `--mutation <type>` | `corrupt` | `corrupt` \| `scale` \| `zero` |
| `--add-tensors <n>` | `2` | Synthetic tensors to add |
| `--remove-tensors <n>` | `1` | Tensors to remove from target |
| `--seed <u64>` | `42` | RNG seed for reproducible runs |
| `--keep-tmp` | off | Keep temp files after run |
| `--verbose` | off | Per-tensor pass/fail output |

Exit code 0 = all checks passed. Non-zero = mismatch found.

---

## Tests

```bash
cargo test
```

Tests cover pack/verify roundtrip, corruption detection, diff/apply (unchanged/modified/added/removed) including delta codecs, safetensors export, pytorch export, GGUF export with magic/version verification, cast format tests (F16 roundtrip, BF16 roundtrip, Q8_0 reconstruction quality, Q4_0/Q4_K block sizing), cast integration tests, sidecar file round-trip (including binary `tokenizer.model`), tokenizer byte-exact round-trip, bad-magic rejection.

---

## What BLAZE-X is Not

BLAZE-X is not an inference engine. It does not run models. It is not a replacement for llama.cpp, vLLM, or any serving stack.

It is a packaging layer. Think of it as what `.tar.gz` is for software releases — a stable, verifiable container that everything else can be built on top of.

---

## Roadmap (community input welcome)

- Sharded GGUF export for large models
- Streaming pack from remote safetensors URLs
- Patch signing / provenance
- Per-layer cast rules (e.g. keep attention in F16, quantise FFN to Q4_K)
- Custom sidecar file injection (embed arbitrary files not in the standard set)

---

## License

Apache 2.0
