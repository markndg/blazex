//! Loaders — import models from existing formats into BXP archives
//!
//! Currently supported:
//!   • SafeTensors (HuggingFace)  — `.safetensors` files (single or sharded)
//!
//! Each loader returns an iterator of (name, dtype, shape, raw_bytes) ready
//! to be fed into ArchiveWriter.

use crate::types::DType;
use anyhow::{bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;

use std::path::{Path, PathBuf};

/// Describes one tensor as returned by a loader
pub struct LoadedTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub raw: Vec<u8>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SafeTensors loader
// ─────────────────────────────────────────────────────────────────────────────

/// Load all tensors from a HuggingFace model directory.
///
/// Handles both single-file (`model.safetensors`) and sharded
/// (`model-00001-of-00003.safetensors`) layouts.
#[allow(dead_code)] // Public hook for embedding / future `blazex pack` wiring.
pub fn load_safetensors_dir<P: AsRef<Path>>(dir: P) -> Result<Vec<LoadedTensor>> {
    let dir = dir.as_ref();
    let mut shard_files: Vec<PathBuf> = Vec::new();

    // Check for sharded index first
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let idx: serde_json::Value = serde_json::from_reader(
            fs::File::open(&index_path).context("opening shard index")?,
        )?;
        // Collect unique shard filenames in order
        if let Some(map) = idx.get("weight_map").and_then(|v| v.as_object()) {
            let mut shards: Vec<String> = map.values()
                .filter_map(|v| v.as_str().map(|s| s.to_owned()))
                .collect();
            shards.sort();
            shards.dedup();
            for s in shards {
                shard_files.push(dir.join(&s));
            }
        }
    }

    // Fall back to single file
    if shard_files.is_empty() {
        let single = dir.join("model.safetensors");
        if single.exists() {
            shard_files.push(single);
        }
    }

    if shard_files.is_empty() {
        bail!("no safetensors files found in {}", dir.display());
    }

    let total_shards = shard_files.len();
    println!("  Found {} shard(s)", total_shards);

    let mut all: Vec<LoadedTensor> = Vec::new();
    for (i, shard) in shard_files.iter().enumerate() {
        println!("  Loading shard {}/{}: {}", i + 1, total_shards, shard.display());
        let mut tensors = load_safetensors_file(shard)?;
        all.append(&mut tensors);
        // Each shard's raw bytes are freed here — only metadata + copied tensor
        // slices remain.  For large models use pack_safetensors_dir_streaming
        // in main.rs which avoids accumulating all tensors at once.
    }
    Ok(all)
}

/// Returns the ordered list of shard paths without loading anything into RAM.
/// Used by the streaming pack path in main.rs.
pub fn shard_paths<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    let mut shard_files: Vec<PathBuf> = Vec::new();
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value = serde_json::from_reader(
            fs::File::open(&index_path).context("opening shard index")?,
        )?;
        let map = index["weight_map"].as_object().context("bad shard index")?;
        let mut shards: Vec<String> = map.values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shards.sort();
        shards.dedup();
        for s in shards {
            shard_files.push(dir.join(&s));
        }
    } else {
        let single = dir.join("model.safetensors");
        if single.exists() {
            shard_files.push(single);
        }
    }
    if shard_files.is_empty() {
        bail!("no safetensors files found in {}", dir.display());
    }
    Ok(shard_files)
}

/// Load a single `.safetensors` file using memory-mapping.
/// The file is mapped read-only — OS handles paging, no full read into RAM.
/// Each tensor slice is copied out individually so callers can free the map.
pub fn load_safetensors_file<P: AsRef<Path>>(path: P) -> Result<Vec<LoadedTensor>> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;

    // SAFETY: The file is read-only and not modified during the map lifetime.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("mmap {}", path.display()))?;
    let bytes: &[u8] = &mmap;

    if bytes.len() < 8 {
        bail!("file too small to be safetensors");
    }

    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    if header_end > bytes.len() {
        bail!("truncated safetensors header");
    }

    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])?;
    let obj = header.as_object().context("safetensors header not an object")?;

    let pb = ProgressBar::new(obj.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    {spinner:.green} [{bar:35}] {pos}/{len} tensors")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut tensors: Vec<LoadedTensor> = Vec::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            pb.inc(1);
            continue;
        }
        let dtype_str = meta["dtype"].as_str().context("missing dtype")?;
        let dtype = DType::from_str(dtype_str)
            .with_context(|| format!("unknown dtype '{dtype_str}' for tensor '{name}'"))?;
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .context("missing shape")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offsets = meta["data_offsets"]
            .as_array()
            .context("missing data_offsets")?;
        let start = offsets[0].as_u64().context("bad offset")? as usize + header_end;
        let end   = offsets[1].as_u64().context("bad offset")? as usize + header_end;
        // Copy the slice — mmap is released when this function returns
        let raw = bytes[start..end].to_vec();
        tensors.push(LoadedTensor { name: name.clone(), dtype, shape, raw });
        pb.inc(1);
    }
    pb.finish_and_clear();

    // mmap is unmapped here — OS reclaims the virtual pages
    Ok(tensors)
}

/// Read config.json from a HuggingFace model directory.
pub fn read_config<P: AsRef<Path>>(dir: P) -> Result<serde_json::Value> {
    let p = dir.as_ref().join("config.json");
    let v: serde_json::Value = serde_json::from_reader(
        fs::File::open(&p).with_context(|| format!("opening {}", p.display()))?,
    )?;
    Ok(v)
}

/// Read tokenizer.json from a HuggingFace model directory as raw bytes.
/// We intentionally do NOT parse the JSON — the verbatim text is stored in
/// the archive and re-emitted unchanged on export so formatting is preserved.
pub fn read_tokenizer<P: AsRef<Path>>(dir: P) -> Option<String> {
    let p = dir.as_ref().join("tokenizer.json");
    if !p.exists() {
        return None;
    }
    fs::read_to_string(&p).ok()
}

/// Read all sidecar files that should be embedded alongside tensors.
///
/// Included when present:
///   tokenizer_config.json  — required by AutoTokenizer.from_pretrained()
///   special_tokens_map.json — BOS/EOS/PAD/UNK/MASK token definitions
///   tokenizer.model         — SentencePiece binary (LLaMA, Mistral, T5, …)
///   generation_config.json  — default generation parameters
///   vocab.json              — BPE vocabulary (GPT-2, RoBERTa, Falcon, …)
///   merges.txt              — BPE merge rules (GPT-2, RoBERTa, Falcon, …)
///
/// Returns (filename, raw_bytes) pairs for all files that exist.
pub fn read_sidecar_files<P: AsRef<Path>>(dir: P) -> Vec<(String, Vec<u8>)> {
    let dir = dir.as_ref();
    let candidates = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ];
    let mut found = Vec::new();
    for name in &candidates {
        let p = dir.join(name);
        if p.exists() {
            match fs::read(&p) {
                Ok(bytes) => found.push((name.to_string(), bytes)),
                Err(e) => eprintln!("  Warning: could not read {name}: {e}"),
            }
        }
    }
    found
}
