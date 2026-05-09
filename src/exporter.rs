//! Exporters — write a BLZ archive back to standard formats
//!
//! Supported targets:
//!   safetensors  — single-file HuggingFace SafeTensors (+ config.json, tokenizer.json)
//!   pytorch      — raw .bin tensors + manifest.json  (load with torch.load)
//!   gguf         — native GGUF v3 (no external tools required)
//!   raw          — one file per tensor, named <n>.bin  (universal)
//!
//! Every export function accepts an optional `CastTarget` that converts
//! weights on the fly before writing.  Pass `None` to export as-is.

use crate::cast::{cast_tensor, CastTarget};
use crate::format::{ArchiveReader, TensorEntry};
use crate::types::DType;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Value};

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct ExportStats {
    /// Which exporter ran (`safetensors`, `pytorch`, `gguf`, `raw`).
    #[allow(dead_code)]
    pub format: &'static str,
    pub tensors: usize,
    pub files_written: usize,
    pub bytes_written: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeTensors export
// ─────────────────────────────────────────────────────────────────────────────

/// Export to a single-file `.safetensors` + side-car files.
pub fn export_safetensors<P: AsRef<Path>>(
    archive: &ArchiveReader,
    out_dir: P,
    cast: Option<CastTarget>,
) -> Result<ExportStats> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let tensors = &archive.header.tensors;
    let pb = progress_bar(tensors.len() as u64, "safetensors");

    // ── Pre-compute cast bytes so header offsets are accurate ──
    let mut cast_data: Vec<(Vec<u8>, DType)> = Vec::with_capacity(tensors.len());
    for te in tensors {
        let (bytes, dtype) = get_bytes(archive, te, cast)?;
        cast_data.push((bytes.into_owned(), dtype));
    }

    // ── Build SafeTensors header ──
    let mut header_map: serde_json::Map<String, Value> = serde_json::Map::new();
    let mut data_cursor: u64 = 0;
    for (te, (ref bytes, dtype)) in tensors.iter().zip(cast_data.iter()) {
        let end = data_cursor + bytes.len() as u64;
        header_map.insert(
            te.name.clone(),
            json!({
                "dtype": dtype.as_str(),
                "shape": te.shape,
                "data_offsets": [data_cursor, end]
            }),
        );
        data_cursor = end;
    }
    let header_json = serde_json::to_string(&Value::Object(header_map))?;
    // SafeTensors requires header length to be a multiple of 8
    let padded_len = (header_json.len() + 7) & !7;
    let padding = padded_len - header_json.len();

    let out_path = out_dir.join("model.safetensors");
    let f = fs::File::create(&out_path)?;
    let mut w = BufWriter::new(f);

    w.write_u64::<LittleEndian>(padded_len as u64)?;
    w.write_all(header_json.as_bytes())?;
    for _ in 0..padding {
        w.write_u8(0x20)?; // space padding
    }

    let mut bytes_written: u64 = 8 + padded_len as u64;
    for (_, (ref bytes, _)) in tensors.iter().zip(cast_data.iter()) {
        w.write_all(bytes)?;
        bytes_written += bytes.len() as u64;
        pb.inc(1);
    }
    w.flush()?;
    pb.finish_and_clear();

    // Side-cars
    write_config(out_dir, &archive.header.model_config)?;
    write_tokenizer(out_dir, archive.header.tokenizer.as_deref())?;
    let n_sidecars = write_sidecars(out_dir, archive)?;

    Ok(ExportStats {
        format: "safetensors",
        tensors: tensors.len(),
        files_written: 1
            + if archive.header.tokenizer.is_some() { 2 } else { 1 }
            + n_sidecars,
        bytes_written,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// PyTorch export (raw .bin + manifest)
// ─────────────────────────────────────────────────────────────────────────────

/// Export tensors as raw little-endian binary files with a JSON manifest.
/// Load in Python:
///
/// ```python
/// import json, numpy as np, torch
/// manifest = json.load(open("manifest.json"))
/// tensors = {}
/// for t in manifest["tensors"]:
///     raw = np.fromfile(t["file"], dtype=np.dtype(t["numpy_dtype"]))
///     tensors[t["name"]] = torch.from_numpy(raw.reshape(t["shape"]))
/// ```
pub fn export_pytorch<P: AsRef<Path>>(
    archive: &ArchiveReader,
    out_dir: P,
    cast: Option<CastTarget>,
) -> Result<ExportStats> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let tensors = &archive.header.tensors;
    let pb = progress_bar(tensors.len() as u64, "pytorch");

    let mut manifest_tensors: Vec<Value> = Vec::new();
    let mut bytes_written: u64 = 0;

    for te in tensors {
        // Sanitise filename: replace "/" with "." (common in HF tensor names)
        let safe_name = te.name.replace('/', ".");
        let fname = format!("{safe_name}.bin");
        let fpath = out_dir.join(&fname);

        // Create parent dirs for nested names
        if let Some(parent) = fpath.parent() {
            fs::create_dir_all(parent)?;
        }

        let (bytes, eff_dtype) = get_bytes(archive, te, cast)?;
        fs::write(&fpath, bytes.as_ref())?;
        bytes_written += bytes.len() as u64;

        manifest_tensors.push(json!({
            "name": te.name,
            "file": fname,
            "dtype": eff_dtype.as_str(),
            "numpy_dtype": dtype_to_numpy(eff_dtype),
            "shape": te.shape,
            "xxh3": format!("{:016x}", te.xxh3),
        }));
        pb.inc(1);
    }
    pb.finish_and_clear();

    let manifest = json!({
        "format": "blz-pytorch-export",
        "version": 1,
        "model_config": archive.header.model_config,
        "tensors": manifest_tensors,
    });
    let mpath = out_dir.join("manifest.json");
    fs::write(&mpath, serde_json::to_vec_pretty(&manifest)?)?;

    write_config(out_dir, &archive.header.model_config)?;
    write_tokenizer(out_dir, archive.header.tokenizer.as_deref())?;
    let n_sidecars = write_sidecars(out_dir, archive)?;

    // Write helper loader script
    let loader_py = include_str!("../scripts/pytorch_loader.py");
    fs::write(out_dir.join("load_model.py"), loader_py)?;

    Ok(ExportStats {
        format: "pytorch",
        tensors: tensors.len(),
        files_written: tensors.len() + 3 + n_sidecars,
        bytes_written,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF export — full native Rust writer
// ─────────────────────────────────────────────────────────────────────────────
//
// Writes a valid GGUF v3 file directly — no Python, no external tools.
//
// Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//
// Layout:
//   [magic 4B] [version u32] [tensor_count u64] [kv_count u64]
//   [metadata kv pairs ...]
//   [tensor info entries ...]
//   <alignment padding to ALIGNMENT>
//   [tensor data blobs, each padded to ALIGNMENT]
//
// We write unquantised weights (F32/F16/BF16 as-is) because quantisation
// requires calibration data and is out of scope for a lossless packager.
// The resulting file is directly loadable by llama.cpp and Ollama.

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGNMENT: u64 = 32; // standard default

/// GGUF value type tags (from gguf.md §GGUFValueType)
#[repr(u32)]
#[allow(dead_code)] // Full enum mirrors the spec; only a subset are written today.
enum GgufType {
    Uint8   = 0,
    Int8    = 1,
    Uint16  = 2,
    Int16   = 3,
    Uint32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    Uint64  = 10,
    Int64   = 11,
    Float64 = 12,
}

/// GGML tensor type tags (from ggml.h enum ggml_type)
#[repr(u32)]
#[derive(Clone, Copy)]
#[allow(dead_code)] // Reserved for dtype→GGML mapping when extending GGUF writer.
enum GgmlType {
    F32  = 0,
    F16  = 1,
    BF16 = 30, // added in GGUF v3
    I8   = 24,
    I16  = 25,
    I32  = 26,
    I64  = 27,
    // Fallback: store unknown dtypes as raw bytes via F32 reinterpretation
    // is avoided — we use F32 only when the source dtype is F32.
}

#[allow(dead_code)]
fn dtype_to_ggml(dtype: DType) -> GgmlType {
    match dtype {
        DType::F32  => GgmlType::F32,
        DType::F16  => GgmlType::F16,
        DType::BF16 => GgmlType::BF16,
        DType::I8   => GgmlType::I8,
        DType::I32  => GgmlType::I32,
        DType::I64  => GgmlType::I64,
        // For types with no direct GGML equivalent, fall back to F32
        // (U8, U16, U32, Bool) — caller should be aware.
        _           => GgmlType::F32,
    }
}

/// Write a GGUF-encoded string: u64 length + non-null-terminated UTF-8 bytes
fn write_gguf_str(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())
}

/// Write a metadata KV pair where the value is a string
fn write_kv_string(w: &mut impl Write, key: &str, value: &str) -> std::io::Result<()> {
    write_gguf_str(w, key)?;
    w.write_u32::<LittleEndian>(GgufType::String as u32)?;
    write_gguf_str(w, value)
}

/// Write a metadata KV pair where the value is a uint32
fn write_kv_u32(w: &mut impl Write, key: &str, value: u32) -> std::io::Result<()> {
    write_gguf_str(w, key)?;
    w.write_u32::<LittleEndian>(GgufType::Uint32 as u32)?;
    w.write_u32::<LittleEndian>(value)
}

/// Write a metadata KV pair where the value is a uint64
#[allow(dead_code)]
fn write_kv_u64(w: &mut impl Write, key: &str, value: u64) -> std::io::Result<()> {
    write_gguf_str(w, key)?;
    w.write_u32::<LittleEndian>(GgufType::Uint64 as u32)?;
    w.write_u64::<LittleEndian>(value)
}

/// Write a metadata KV pair where the value is a float32
fn write_kv_f32(w: &mut impl Write, key: &str, value: f32) -> std::io::Result<()> {
    write_gguf_str(w, key)?;
    w.write_u32::<LittleEndian>(GgufType::Float32 as u32)?;
    w.write_f32::<LittleEndian>(value)
}

/// Write a metadata KV pair where the value is an array of strings
fn write_kv_str_array(w: &mut impl Write, key: &str, values: &[String]) -> std::io::Result<()> {
    write_gguf_str(w, key)?;
    w.write_u32::<LittleEndian>(GgufType::Array as u32)?;
    // Array element type = String
    w.write_u32::<LittleEndian>(GgufType::String as u32)?;
    w.write_u64::<LittleEndian>(values.len() as u64)?;
    for v in values {
        write_gguf_str(w, v)?;
    }
    Ok(())
}

/// Pad writer to next multiple of GGUF_ALIGNMENT with zero bytes.
fn write_padding(w: &mut impl Write, current_pos: u64) -> std::io::Result<u64> {
    let rem = current_pos % GGUF_ALIGNMENT;
    if rem == 0 {
        return Ok(0);
    }
    let pad = GGUF_ALIGNMENT - rem;
    let zeros = vec![0u8; pad as usize];
    w.write_all(&zeros)?;
    Ok(pad)
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF tensor name mapping
// Canonical reference: gguf-py/gguf/tensor_mapping.py in llama.cpp repo
// ─────────────────────────────────────────────────────────────────────────────

/// Map a HF tensor name → GGUF standard name.
/// Returns None for tensors that should be skipped in GGUF output
/// (rotary buffers, vision-tower weights, etc.)
fn hf_name_to_gguf(name: &str, arch: &str) -> Option<String> {
    if name.contains("rotary_emb.inv_freq")
        || name.contains("vision_tower")
        || name.contains("visual_projection")
        || name.contains("multi_modal_projector")
    {
        return None;
    }
    let result = match arch {
        "gpt2" | "gpt_refact"          => map_gpt2(name),
        "gpt_neox" | "pythia"
            | "stablelm"               => map_gptneox(name),
        "falcon" | "refinedweb"        => map_falcon(name),
        "mpt"                          => map_mpt(name),
        "bloom"                        => map_bloom(name),
        "gemma" | "gemma2"             => map_gemma(name),
        "phi" | "phi-msft"             => map_phi2(name),
        "phi3"                         => map_phi3(name),
        "qwen"                         => map_qwen(name),
        "qwen2" | "qwen2_moe"          => map_qwen2(name),
        "starcoder2"                   => map_starcoder2(name),
        "internlm2"                    => map_internlm2(name),
        "chatglm"                      => map_chatglm(name),
        "command-r" | "cohere"         => map_command_r(name),
        "deepseek" | "deepseek2"       => map_deepseek(name),
        // llama, llama2, llama3, mistral, mixtral, yi, solar, vicuna,
        // nemotron, olmo, granite, qwen2.5, and most fine-tunes
        _                              => map_llama(name),
    };
    Some(result)
}

fn layer_rewrite(s: &str, prefix: &str) -> String {
    if let Some(rest) = s.strip_prefix(prefix) {
        if let Some(dot) = rest.find('.') {
            return format!("blk.{}.{}", &rest[..dot], &rest[dot+1..]);
        }
    }
    s.to_owned()
}

fn map_llama(name: &str) -> String {
    let s = name
        .replace("model.embed_tokens.weight",            "token_embd.weight")
        .replace("model.norm.weight",                    "output_norm.weight")
        .replace("lm_head.weight",                       "output.weight")
        .replace(".self_attn.q_proj.weight",             ".attn_q.weight")
        .replace(".self_attn.k_proj.weight",             ".attn_k.weight")
        .replace(".self_attn.v_proj.weight",             ".attn_v.weight")
        .replace(".self_attn.o_proj.weight",             ".attn_output.weight")
        .replace(".self_attn.q_proj.bias",               ".attn_q.bias")
        .replace(".self_attn.k_proj.bias",               ".attn_k.bias")
        .replace(".self_attn.v_proj.bias",               ".attn_v.bias")
        .replace(".self_attn.o_proj.bias",               ".attn_output.bias")
        .replace(".self_attn.qkv_proj.weight",           ".attn_qkv.weight")
        .replace(".block_sparse_moe.gate.weight",        ".ffn_gate_inp.weight")
        .replace(".mlp.gate_proj.weight",                ".ffn_gate.weight")
        .replace(".mlp.up_proj.weight",                  ".ffn_up.weight")
        .replace(".mlp.down_proj.weight",                ".ffn_down.weight")
        .replace(".mlp.gate_proj.bias",                  ".ffn_gate.bias")
        .replace(".mlp.up_proj.bias",                    ".ffn_up.bias")
        .replace(".mlp.down_proj.bias",                  ".ffn_down.bias")
        .replace(".input_layernorm.weight",              ".attn_norm.weight")
        .replace(".post_attention_layernorm.weight",     ".ffn_norm.weight")
        .replace(".input_layernorm.bias",                ".attn_norm.bias")
        .replace(".post_attention_layernorm.bias",       ".ffn_norm.bias");
    layer_rewrite(&s, "model.layers.")
}

fn map_gpt2(name: &str) -> String {
    let s = name
        .replace("transformer.wte.weight",               "token_embd.weight")
        .replace("transformer.wpe.weight",               "position_embd.weight")
        .replace("transformer.ln_f.weight",              "output_norm.weight")
        .replace("transformer.ln_f.bias",                "output_norm.bias")
        .replace("lm_head.weight",                       "output.weight")
        .replace(".attn.c_attn.weight",                  ".attn_qkv.weight")
        .replace(".attn.c_attn.bias",                    ".attn_qkv.bias")
        .replace(".attn.c_proj.weight",                  ".attn_output.weight")
        .replace(".attn.c_proj.bias",                    ".attn_output.bias")
        .replace(".mlp.c_fc.weight",                     ".ffn_up.weight")
        .replace(".mlp.c_fc.bias",                       ".ffn_up.bias")
        .replace(".mlp.c_proj.weight",                   ".ffn_down.weight")
        .replace(".mlp.c_proj.bias",                     ".ffn_down.bias")
        .replace(".ln_1.weight",                         ".attn_norm.weight")
        .replace(".ln_1.bias",                           ".attn_norm.bias")
        .replace(".ln_2.weight",                         ".ffn_norm.weight")
        .replace(".ln_2.bias",                           ".ffn_norm.bias");
    layer_rewrite(&s, "transformer.h.")
}

fn map_gptneox(name: &str) -> String {
    let s = name
        .replace("gpt_neox.embed_in.weight",             "token_embd.weight")
        .replace("gpt_neox.final_layer_norm.weight",     "output_norm.weight")
        .replace("gpt_neox.final_layer_norm.bias",       "output_norm.bias")
        .replace("embed_out.weight",                     "output.weight")
        .replace(".attention.query_key_value.weight",    ".attn_qkv.weight")
        .replace(".attention.query_key_value.bias",      ".attn_qkv.bias")
        .replace(".attention.dense.weight",              ".attn_output.weight")
        .replace(".attention.dense.bias",                ".attn_output.bias")
        .replace(".mlp.dense_h_to_4h.weight",            ".ffn_up.weight")
        .replace(".mlp.dense_h_to_4h.bias",              ".ffn_up.bias")
        .replace(".mlp.dense_4h_to_h.weight",            ".ffn_down.weight")
        .replace(".mlp.dense_4h_to_h.bias",              ".ffn_down.bias")
        .replace(".input_layernorm.weight",              ".attn_norm.weight")
        .replace(".input_layernorm.bias",                ".attn_norm.bias")
        .replace(".post_attention_layernorm.weight",     ".ffn_norm.weight")
        .replace(".post_attention_layernorm.bias",       ".ffn_norm.bias");
    layer_rewrite(&s, "gpt_neox.layers.")
}

fn map_falcon(name: &str) -> String {
    let s = name
        .replace("transformer.word_embeddings.weight",     "token_embd.weight")
        .replace("transformer.ln_f.weight",                "output_norm.weight")
        .replace("transformer.ln_f.bias",                  "output_norm.bias")
        .replace("lm_head.weight",                         "output.weight")
        .replace(".self_attention.query_key_value.weight", ".attn_qkv.weight")
        .replace(".self_attention.query_key_value.bias",   ".attn_qkv.bias")
        .replace(".self_attention.dense.weight",           ".attn_output.weight")
        .replace(".self_attention.dense.bias",             ".attn_output.bias")
        .replace(".mlp.dense_h_to_4h.weight",              ".ffn_up.weight")
        .replace(".mlp.dense_h_to_4h.bias",               ".ffn_up.bias")
        .replace(".mlp.dense_4h_to_h.weight",              ".ffn_down.weight")
        .replace(".mlp.dense_4h_to_h.bias",               ".ffn_down.bias")
        .replace(".ln_attn.weight",                        ".attn_norm.weight")
        .replace(".ln_attn.bias",                          ".attn_norm.bias")
        .replace(".ln_mlp.weight",                         ".attn_norm_2.weight")
        .replace(".ln_mlp.bias",                           ".attn_norm_2.bias")
        .replace(".input_layernorm.weight",                ".attn_norm.weight")
        .replace(".input_layernorm.bias",                  ".attn_norm.bias");
    layer_rewrite(&s, "transformer.h.")
}

fn map_mpt(name: &str) -> String {
    let s = name
        .replace("transformer.wte.weight",               "token_embd.weight")
        .replace("transformer.norm_f.weight",            "output_norm.weight")
        .replace("transformer.norm_f.bias",              "output_norm.bias")
        .replace(".attn.Wqkv.weight",                    ".attn_qkv.weight")
        .replace(".attn.out_proj.weight",                ".attn_output.weight")
        .replace(".ffn.up_proj.weight",                  ".ffn_up.weight")
        .replace(".ffn.down_proj.weight",                ".ffn_down.weight")
        .replace(".norm_1.weight",                       ".attn_norm.weight")
        .replace(".norm_1.bias",                         ".attn_norm.bias")
        .replace(".norm_2.weight",                       ".ffn_norm.weight")
        .replace(".norm_2.bias",                         ".ffn_norm.bias");
    layer_rewrite(&s, "transformer.blocks.")
}

fn map_bloom(name: &str) -> String {
    let s = name
        .replace("transformer.word_embeddings.weight",           "token_embd.weight")
        .replace("transformer.word_embeddings_layernorm.weight", "token_embd_norm.weight")
        .replace("transformer.word_embeddings_layernorm.bias",   "token_embd_norm.bias")
        .replace("transformer.ln_f.weight",                      "output_norm.weight")
        .replace("transformer.ln_f.bias",                        "output_norm.bias")
        .replace("lm_head.weight",                               "output.weight")
        .replace(".self_attention.query_key_value.weight",       ".attn_qkv.weight")
        .replace(".self_attention.query_key_value.bias",         ".attn_qkv.bias")
        .replace(".self_attention.dense.weight",                 ".attn_output.weight")
        .replace(".self_attention.dense.bias",                   ".attn_output.bias")
        .replace(".mlp.dense_h_to_4h.weight",                   ".ffn_up.weight")
        .replace(".mlp.dense_h_to_4h.bias",                     ".ffn_up.bias")
        .replace(".mlp.dense_4h_to_h.weight",                   ".ffn_down.weight")
        .replace(".mlp.dense_4h_to_h.bias",                     ".ffn_down.bias")
        .replace(".input_layernorm.weight",                      ".attn_norm.weight")
        .replace(".input_layernorm.bias",                        ".attn_norm.bias")
        .replace(".post_attention_layernorm.weight",             ".ffn_norm.weight")
        .replace(".post_attention_layernorm.bias",               ".ffn_norm.bias");
    layer_rewrite(&s, "transformer.h.")
}

fn map_gemma(name: &str) -> String {
    // Gemma shares LLaMA layout but Gemma 2 adds extra norms
    let s = map_llama(name)
        .replace(".pre_feedforward_layernorm.weight",    ".ffn_pre_norm.weight")
        .replace(".post_feedforward_layernorm.weight",   ".ffn_post_norm.weight")
        .replace(".post_attention_layernorm.weight",     ".attn_post_norm.weight");
    s
}

fn map_phi2(name: &str) -> String {
    let s = name
        .replace("model.embed_tokens.weight",            "token_embd.weight")
        .replace("model.final_layernorm.weight",         "output_norm.weight")
        .replace("model.final_layernorm.bias",           "output_norm.bias")
        .replace("lm_head.weight",                       "output.weight")
        .replace("lm_head.bias",                         "output.bias")
        .replace(".self_attn.q_proj.weight",             ".attn_q.weight")
        .replace(".self_attn.k_proj.weight",             ".attn_k.weight")
        .replace(".self_attn.v_proj.weight",             ".attn_v.weight")
        .replace(".self_attn.dense.weight",              ".attn_output.weight")
        .replace(".self_attn.q_proj.bias",               ".attn_q.bias")
        .replace(".self_attn.k_proj.bias",               ".attn_k.bias")
        .replace(".self_attn.v_proj.bias",               ".attn_v.bias")
        .replace(".self_attn.dense.bias",                ".attn_output.bias")
        .replace(".mlp.fc1.weight",                      ".ffn_up.weight")
        .replace(".mlp.fc1.bias",                        ".ffn_up.bias")
        .replace(".mlp.fc2.weight",                      ".ffn_down.weight")
        .replace(".mlp.fc2.bias",                        ".ffn_down.bias")
        .replace(".input_layernorm.weight",              ".attn_norm.weight")
        .replace(".input_layernorm.bias",                ".attn_norm.bias");
    layer_rewrite(&s, "model.layers.")
}

fn map_phi3(name: &str) -> String {
    let s = name
        .replace("model.embed_tokens.weight",            "token_embd.weight")
        .replace("model.norm.weight",                    "output_norm.weight")
        .replace("lm_head.weight",                       "output.weight")
        .replace(".self_attn.qkv_proj.weight",           ".attn_qkv.weight")
        .replace(".self_attn.o_proj.weight",             ".attn_output.weight")
        .replace(".mlp.gate_up_proj.weight",             ".ffn_gate_up.weight")
        .replace(".mlp.down_proj.weight",                ".ffn_down.weight")
        .replace(".input_layernorm.weight",              ".attn_norm.weight")
        .replace(".post_attention_layernorm.weight",     ".ffn_norm.weight");
    layer_rewrite(&s, "model.layers.")
}

fn map_qwen(name: &str) -> String {
    let s = name
        .replace("transformer.wte.weight",               "token_embd.weight")
        .replace("transformer.ln_f.weight",              "output_norm.weight")
        .replace("lm_head.weight",                       "output.weight")
        .replace(".attn.c_attn.weight",                  ".attn_qkv.weight")
        .replace(".attn.c_attn.bias",                    ".attn_qkv.bias")
        .replace(".attn.c_proj.weight",                  ".attn_output.weight")
        .replace(".mlp.w1.weight",                       ".ffn_gate.weight")
        .replace(".mlp.w2.weight",                       ".ffn_down.weight")
        .replace(".mlp.c_proj.weight",                   ".ffn_up.weight")
        .replace(".ln_1.weight",                         ".attn_norm.weight")
        .replace(".ln_2.weight",                         ".ffn_norm.weight");
    layer_rewrite(&s, "transformer.h.")
}

fn map_qwen2(name: &str) -> String { map_llama(name) }

fn map_starcoder2(name: &str) -> String {
    let s = name
        .replace("model.embed_tokens.weight",            "token_embd.weight")
        .replace("model.norm.weight",                    "output_norm.weight")
        .replace("model.norm.bias",                      "output_norm.bias")
        .replace("lm_head.weight",                       "output.weight")
        .replace(".self_attn.q_proj.weight",             ".attn_q.weight")
        .replace(".self_attn.k_proj.weight",             ".attn_k.weight")
        .replace(".self_attn.v_proj.weight",             ".attn_v.weight")
        .replace(".self_attn.o_proj.weight",             ".attn_output.weight")
        .replace(".self_attn.q_proj.bias",               ".attn_q.bias")
        .replace(".self_attn.k_proj.bias",               ".attn_k.bias")
        .replace(".self_attn.v_proj.bias",               ".attn_v.bias")
        .replace(".self_attn.o_proj.bias",               ".attn_output.bias")
        .replace(".mlp.c_fc.weight",                     ".ffn_up.weight")
        .replace(".mlp.c_fc.bias",                       ".ffn_up.bias")
        .replace(".mlp.c_proj.weight",                   ".ffn_down.weight")
        .replace(".mlp.c_proj.bias",                     ".ffn_down.bias")
        .replace(".input_layernorm.weight",              ".attn_norm.weight")
        .replace(".input_layernorm.bias",                ".attn_norm.bias")
        .replace(".post_attention_layernorm.weight",     ".ffn_norm.weight")
        .replace(".post_attention_layernorm.bias",       ".ffn_norm.bias");
    layer_rewrite(&s, "model.layers.")
}

fn map_internlm2(name: &str) -> String {
    let s = name
        .replace("model.tok_embeddings.weight",          "token_embd.weight")
        .replace("model.norm.weight",                    "output_norm.weight")
        .replace("output.weight",                        "output.weight")
        .replace(".attention.wqkv.weight",               ".attn_qkv.weight")
        .replace(".attention.wo.weight",                 ".attn_output.weight")
        .replace(".feed_forward.w1.weight",              ".ffn_gate.weight")
        .replace(".feed_forward.w2.weight",              ".ffn_down.weight")
        .replace(".feed_forward.w3.weight",              ".ffn_up.weight")
        .replace(".attention_norm.weight",               ".attn_norm.weight")
        .replace(".ffn_norm.weight",                     ".ffn_norm.weight");
    layer_rewrite(&s, "model.layers.")
}

fn map_chatglm(name: &str) -> String {
    let s = name
        .replace("transformer.embedding.word_embeddings.weight", "token_embd.weight")
        .replace("transformer.encoder.final_layernorm.weight",   "output_norm.weight")
        .replace("transformer.encoder.final_layernorm.bias",     "output_norm.bias")
        .replace("transformer.output_layer.weight",              "output.weight")
        .replace(".self_attention.query_key_value.weight",       ".attn_qkv.weight")
        .replace(".self_attention.query_key_value.bias",         ".attn_qkv.bias")
        .replace(".self_attention.dense.weight",                 ".attn_output.weight")
        .replace(".self_attention.dense.bias",                   ".attn_output.bias")
        .replace(".mlp.dense_h_to_4h.weight",                   ".ffn_up.weight")
        .replace(".mlp.dense_4h_to_h.weight",                   ".ffn_down.weight")
        .replace(".input_layernorm.weight",                      ".attn_norm.weight")
        .replace(".input_layernorm.bias",                        ".attn_norm.bias")
        .replace(".post_attention_layernorm.weight",             ".ffn_norm.weight")
        .replace(".post_attention_layernorm.bias",               ".ffn_norm.bias");
    layer_rewrite(&s, "transformer.encoder.layers.")
}

fn map_command_r(name: &str) -> String { map_llama(name) }

fn map_deepseek(name: &str) -> String {
    let s = name
        .replace("model.embed_tokens.weight",              "token_embd.weight")
        .replace("model.norm.weight",                      "output_norm.weight")
        .replace("lm_head.weight",                         "output.weight")
        .replace(".self_attn.q_proj.weight",               ".attn_q.weight")
        .replace(".self_attn.q_a_proj.weight",             ".attn_q_a.weight")
        .replace(".self_attn.q_b_proj.weight",             ".attn_q_b.weight")
        .replace(".self_attn.kv_a_proj_with_mqa.weight",   ".attn_kv_a_mqa.weight")
        .replace(".self_attn.kv_b_proj.weight",            ".attn_kv_b.weight")
        .replace(".self_attn.o_proj.weight",               ".attn_output.weight")
        .replace(".self_attn.q_a_layernorm.weight",        ".attn_q_a_norm.weight")
        .replace(".self_attn.kv_a_layernorm.weight",       ".attn_kv_a_norm.weight")
        .replace(".mlp.gate.weight",                       ".ffn_gate_inp.weight")
        .replace(".mlp.shared_expert_gate.weight",         ".ffn_gate_inp_shexp.weight")
        .replace(".mlp.gate_proj.weight",                  ".ffn_gate.weight")
        .replace(".mlp.up_proj.weight",                    ".ffn_up.weight")
        .replace(".mlp.down_proj.weight",                  ".ffn_down.weight")
        .replace(".input_layernorm.weight",                ".attn_norm.weight")
        .replace(".post_attention_layernorm.weight",       ".ffn_norm.weight");
    layer_rewrite(&s, "model.layers.")
}



/// Extract architecture metadata from config.json for GGUF metadata section.
/// Returns (arch_name, kv_pairs) where kv_pairs are (key, serialised_value) tuples.
fn extract_arch_metadata(config: &Value) -> (String, Vec<(String, Value)>) {
    let arch = config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_owned();

    let mut kvs: Vec<(String, Value)> = Vec::new();

    // Map standard HF config keys to GGUF metadata keys
    let mappings: &[(&str, &str)] = &[
        ("hidden_size",              "embedding_length"),
        ("num_hidden_layers",        "block_count"),
        ("num_attention_heads",      "attention.head_count"),
        ("num_key_value_heads",      "attention.head_count_kv"),
        ("intermediate_size",        "feed_forward_length"),
        ("max_position_embeddings",  "context_length"),
        ("vocab_size",               "vocab_size"),
        ("rms_norm_eps",             "attention.layer_norm_rms_epsilon"),
        ("rope_theta",               "rope.freq_base"),
        ("sliding_window",           "attention.sliding_window"),
    ];

    for (hf_key, gguf_suffix) in mappings {
        if let Some(v) = config.get(*hf_key) {
            kvs.push((format!("{arch}.{gguf_suffix}"), v.clone()));
        }
    }

    (arch, kvs)
}

/// Build the GGUF metadata KV section and return the serialised bytes + kv_count.
fn build_metadata(archive: &ArchiveReader) -> anyhow::Result<(Vec<u8>, u64)> {
    let mut buf: Vec<u8> = Vec::new();
    let mut kv_count: u64 = 0;

    let (arch, arch_kvs) = extract_arch_metadata(&archive.header.model_config);

    // general.* keys
    write_kv_string(&mut buf, "general.architecture", &arch)?;
    kv_count += 1;

    if let Some(name) = archive.header.model_config.get("_name_or_path").and_then(|v| v.as_str()) {
        write_kv_string(&mut buf, "general.name", name)?;
        kv_count += 1;
    }

    write_kv_string(&mut buf, "general.file_type", "1")?; // 1 = F16 (unquantised passthrough)
    kv_count += 1;

    // Architecture-specific keys from config
    for (key, val) in &arch_kvs {
        if let Some(n) = val.as_u64() {
            write_kv_u32(&mut buf, key, n as u32)?;
            kv_count += 1;
        } else if let Some(f) = val.as_f64() {
            write_kv_f32(&mut buf, key, f as f32)?;
            kv_count += 1;
        }
    }

    // Tokenizer metadata from embedded tokenizer.json
    if let Some(tok_raw) = &archive.header.tokenizer {
        let tok: serde_json::Value = serde_json::from_str(tok_raw).unwrap_or(serde_json::Value::Null);
        let tok = &tok;
        // BPE / sentencepiece model type
        let tok_type = tok.get("model")
            .and_then(|m| m.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("BPE");
        // GGUF tokenizer model: "gpt2" for BPE, "llama" for SentencePiece
        let gguf_tok_model = match tok_type.to_uppercase().as_str() {
            "BPE" => "gpt2",
            "UNIGRAM" | "SENTENCEPIECE" => "llama",
            _ => "gpt2",
        };
        write_kv_string(&mut buf, "tokenizer.ggml.model", gguf_tok_model)?;
        kv_count += 1;

        // Vocabulary tokens
        if let Some(vocab) = tok.get("model").and_then(|m| m.get("vocab")) {
            if let Some(obj) = vocab.as_object() {
                // Sort by token id
                let mut pairs: Vec<(u64, String)> = obj.iter()
                    .filter_map(|(token, id)| id.as_u64().map(|i| (i, token.clone())))
                    .collect();
                pairs.sort_by_key(|(id, _)| *id);
                let tokens: Vec<String> = pairs.into_iter().map(|(_, t)| t).collect();
                write_kv_str_array(&mut buf, "tokenizer.ggml.tokens", &tokens)?;
                kv_count += 1;
            }
        }

        // Special token IDs
        let special_ids: &[(&str, &str)] = &[
            ("bos_token_id", "tokenizer.ggml.bos_token_id"),
            ("eos_token_id", "tokenizer.ggml.eos_token_id"),
            ("unk_token_id", "tokenizer.ggml.unknown_token_id"),
            ("pad_token_id", "tokenizer.ggml.padding_token_id"),
        ];
        // These come from the outer tokenizer config or model config
        for (cfg_key, gguf_key) in special_ids {
            let val = tok.get(*cfg_key)
                .or_else(|| archive.header.model_config.get(*cfg_key));
            if let Some(id) = val.and_then(|v| v.as_u64()) {
                write_kv_u32(&mut buf, gguf_key, id as u32)?;
                kv_count += 1;
            }
        }
    }

    Ok((buf, kv_count))
}

/// Write a complete GGUF v3 file to `out_path`.
pub fn export_gguf<P: AsRef<Path>>(archive: &ArchiveReader, out_path: P, cast: Option<CastTarget>) -> Result<ExportStats> {

    // Accept either a directory (like safetensors/pytorch) or an explicit .gguf
    // file path. If no .gguf extension, treat as a directory and write model.gguf inside it.
    let out_path_raw = out_path.as_ref();
    let gguf_file: std::path::PathBuf =
        if out_path_raw.extension().and_then(|e| e.to_str()) == Some("gguf") {
            out_path_raw.to_path_buf()
        } else {
            out_path_raw.join("model.gguf")
        };
    let out_path = gguf_file.as_path();
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tensors = &archive.header.tensors;
    let pb = progress_bar(tensors.len() as u64, "gguf");

    // ── Phase 1: build metadata bytes ──
    let (meta_bytes, kv_count) = build_metadata(archive)?;

    // ── Phase 2: pre-compute cast bytes and tensor data offsets ──
    // Cast all tensors first so we know the exact output sizes for offset calculation.
    let mut cast_tensors: Vec<(Vec<u8>, CastTarget)> = Vec::with_capacity(tensors.len());
    for te in tensors {
        let src_cast = cast.unwrap_or(match te.dtype {
            crate::types::DType::F32  => CastTarget::F32,
            crate::types::DType::F16  => CastTarget::F16,
            crate::types::DType::BF16 => CastTarget::BF16,
            // Integer/bool tensors — keep as F32 in GGUF (norm weights etc.)
            _                         => CastTarget::F32,
        });
        let raw = archive.tensor_bytes(&te.name)?;
        let bytes = cast_tensor(raw, te.dtype, src_cast, te.element_count())?;
        cast_tensors.push((bytes, src_cast));
    }

    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut data_cursor: u64 = 0;
    for (bytes, _) in &cast_tensors {
        offsets.push(data_cursor);
        data_cursor += bytes.len() as u64;
        let rem = data_cursor % GGUF_ALIGNMENT;
        if rem != 0 {
            data_cursor += GGUF_ALIGNMENT - rem;
        }
    }

    // ── Phase 3: open file and write header ──
    let f = fs::File::create(out_path)
        .with_context(|| format!("creating {}", out_path.display()))?;
    let mut w = BufWriter::new(f);

    w.write_u32::<LittleEndian>(GGUF_MAGIC)?;
    w.write_u32::<LittleEndian>(GGUF_VERSION)?;
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(kv_count)?;

    // ── Phase 4: write metadata KV section ──
    w.write_all(&meta_bytes)?;

    // ── Phase 5: write tensor info array ──
    for (te, &offset) in tensors.iter().zip(offsets.iter()) {
        let gguf_name = match hf_name_to_gguf(&te.name,
            archive.header.model_config.get("model_type")
                .and_then(|v| v.as_str()).unwrap_or("llama"))
        {
            Some(n) => n,
            None => continue, // skip unmapped tensors (vision, rotary buffers)
        };
        write_gguf_str(&mut w, &gguf_name)?;

        // ndims and dims — GGUF stores shape reversed (innermost first)
        let ndims = te.shape.len() as u32;
        w.write_u32::<LittleEndian>(ndims)?;
        for &d in te.shape.iter().rev() {
            w.write_u64::<LittleEndian>(d as u64)?;
        }

        let ggml_type_val = cast_tensors[tensors.iter().position(|t| t.name == te.name).unwrap()].1.ggml_type_tag();
        w.write_u32::<LittleEndian>(ggml_type_val)?;
        w.write_u64::<LittleEndian>(offset)?;

        pb.inc(1);
    }

    // ── Phase 6: alignment padding before data region ──
    // Compute current write position: header + meta + tensor_info
    // We can't seek a BufWriter directly, so we track it manually.
    // header = 4+4+8+8 = 24 bytes
    // meta   = meta_bytes.len()
    // tensor_info = sum of each entry's serialised size
    let mut tensor_info_size: u64 = 0;
    for te in tensors {
        let gguf_name = match hf_name_to_gguf(&te.name,
            archive.header.model_config.get("model_type")
                .and_then(|v| v.as_str()).unwrap_or("llama"))
        {
            Some(n) => n,
            None => continue, // skip unmapped tensors
        };
        // 8 (name len) + name.len() + 4 (ndims) + ndims*8 (dims) + 4 (type) + 8 (offset)
        tensor_info_size += 8 + gguf_name.len() as u64 + 4
            + te.shape.len() as u64 * 8 + 4 + 8;
    }
    let pre_data_pos: u64 = 24 + meta_bytes.len() as u64 + tensor_info_size;
    let pad = write_padding(&mut w, pre_data_pos)?;

    // ── Phase 7: write tensor data ──
    let mut bytes_written: u64 = 0;
    for (cast_bytes, _) in &cast_tensors {
        w.write_all(cast_bytes)?;
        bytes_written += cast_bytes.len() as u64;
        let rem = cast_bytes.len() as u64 % GGUF_ALIGNMENT;
        if rem != 0 {
            let pad_len = (GGUF_ALIGNMENT - rem) as usize;
            w.write_all(&vec![0u8; pad_len])?;
        }
    }
    w.flush()?;
    pb.finish_and_clear();

    // Write config.json, tokenizer.json, and all sidecar files alongside the
    // .gguf file so the output directory is a complete model directory.
    // The .gguf itself is self-contained for llama.cpp/Ollama, but the
    // accompanying files are needed for HF transformers, Hub uploads, etc.
    let out_dir = out_path.parent().unwrap_or(std::path::Path::new("."));
    write_config(out_dir, &archive.header.model_config)?;
    write_tokenizer(out_dir, archive.header.tokenizer.as_deref())?;
    let n_sidecars = write_sidecars(out_dir, archive)?;

    let total = 24 + meta_bytes.len() as u64 + tensor_info_size + pad + bytes_written;
    Ok(ExportStats {
        format: "gguf",
        tensors: tensors.len(),
        files_written: 1 + 1 + if archive.header.tokenizer.is_some() { 1 } else { 0 } + n_sidecars,
        bytes_written: total,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw export — one .bin per tensor
// ─────────────────────────────────────────────────────────────────────────────

pub fn export_raw<P: AsRef<Path>>(archive: &ArchiveReader, out_dir: P, cast: Option<CastTarget>) -> Result<ExportStats> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let tensors = &archive.header.tensors;
    let pb = progress_bar(tensors.len() as u64, "raw");
    let mut bytes_written: u64 = 0;

    let mut index: Vec<Value> = Vec::with_capacity(tensors.len());
    for te in tensors {
        let safe_name = te.name.replace('/', ".");
        let (bytes, eff_dtype) = get_bytes(archive, te, cast)?;
        fs::write(out_dir.join(format!("{safe_name}.bin")), bytes.as_ref())?;
        bytes_written += bytes.len() as u64;
        index.push(json!({
            "name": te.name,
            "file": format!("{}.bin", te.name.replace('/', ".")),
            "dtype": eff_dtype.as_str(),
            "shape": te.shape,
            "xxh3": format!("{:016x}", te.xxh3),
        }));
        pb.inc(1);
    }
    pb.finish_and_clear();
    fs::write(
        out_dir.join("index.json"),
        serde_json::to_vec_pretty(&json!({ "tensors": index }))?,
    )?;
    let n_sidecars = write_sidecars(out_dir, archive)?;

    Ok(ExportStats {
        format: "raw",
        tensors: tensors.len(),
        files_written: tensors.len() + 1 + n_sidecars,
        bytes_written,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Get tensor bytes, applying an optional cast.  Returns (bytes, effective_dtype).
/// When cast is None this is a zero-copy borrow from the mmap.
fn get_bytes<'a>(
    archive: &'a ArchiveReader,
    entry: &TensorEntry,
    cast: Option<CastTarget>,
) -> Result<(std::borrow::Cow<'a, [u8]>, DType)> {
    let raw = archive.tensor_bytes(&entry.name)?;
    match cast {
        None => Ok((std::borrow::Cow::Borrowed(raw), entry.dtype)),
        Some(target) => {
            let n = entry.element_count();
            let out = cast_tensor(raw, entry.dtype, target, n)?;
            Ok((std::borrow::Cow::Owned(out), target.output_dtype()))
        }
    }
}

fn write_config(dir: &Path, config: &Value) -> Result<()> {
    fs::write(dir.join("config.json"), serde_json::to_vec_pretty(config)?)?;
    Ok(())
}

fn write_tokenizer(dir: &Path, tok: Option<&str>) -> Result<()> {
    if let Some(raw) = tok {
        fs::write(dir.join("tokenizer.json"), raw.as_bytes())?;
    }
    Ok(())
}

/// Write all embedded sidecar files to the output directory.
/// Returns the number of files written.
fn write_sidecars(dir: &Path, archive: &ArchiveReader) -> Result<usize> {
    let mut n = 0usize;
    for sidecar in &archive.header.sidecar_files {
        let bytes = sidecar.decode()
            .map_err(|e| anyhow::anyhow!("base64 decode failed for {}: {e}", sidecar.filename))?;
        fs::write(dir.join(&sidecar.filename), &bytes)?;
        n += 1;
    }
    Ok(n)
}

fn progress_bar(len: u64, fmt: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );
    pb.set_message(fmt.to_owned());
    pb
}

fn dtype_to_numpy(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I8 => "int8",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::Bool => "bool",
    }
}
