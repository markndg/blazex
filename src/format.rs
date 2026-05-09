//! BXP Archive Format — stable, versioned, self-describing
//!
//! Layout:
//!   [MAGIC 8B][VERSION 4B][HEADER_LEN 8B][HEADER JSON][TENSOR_DATA...]
//!
//! Every tensor is stored as a named blob of raw bytes (little-endian) preceded
//! by an 8-byte length prefix.  The header carries all metadata needed to
//! reconstruct shape / dtype / ordering without reading the data.
//!
//! Nothing is compressed here.  This is the stable substrate on which
//! diff/patch, export, and verification operate.

use crate::types::{DType};
use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use xxhash_rust::xxh3::xxh3_64;

/// 8-byte magic: "BLAZXPK\n"
pub const MAGIC: u64 = 0x0A4B50585A414C42;
/// Format version — bump when binary layout changes
pub const FORMAT_VERSION: u32 = 1;

/// A sidecar file embedded verbatim in the archive.
/// Content is base64-encoded to survive JSON embedding safely for both text
/// and binary files (e.g. tokenizer.model SentencePiece binary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarFile {
    /// Original filename (e.g. "tokenizer_config.json", "tokenizer.model")
    pub filename: String,
    /// Raw file bytes, base64-encoded
    pub content_b64: String,
}

impl SidecarFile {
    pub fn from_bytes(filename: &str, bytes: &[u8]) -> Self {
        use base64::{engine::general_purpose::STANDARD, Engine};
        Self {
            filename: filename.to_owned(),
            content_b64: STANDARD.encode(bytes),
        }
    }

    pub fn decode(&self) -> Result<Vec<u8>, base64::DecodeError> {
        use base64::{engine::general_purpose::STANDARD, Engine};
        STANDARD.decode(&self.content_b64)
    }
}

/// Stored in the JSON header at the start of every archive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveHeader {
    pub version: u32,
    pub created_at: String,
    /// Free-form model metadata (config.json contents if available)
    pub model_config: serde_json::Value,
    /// tokenizer.json if embedded — stored as the verbatim source bytes to preserve
    /// exact formatting, number representation, and key order.
    pub tokenizer: Option<String>,
    /// Additional sidecar files from the source model directory.
    /// Includes: tokenizer_config.json, special_tokens_map.json, tokenizer.model,
    /// generation_config.json, vocab.json, merges.txt (whichever are present).
    #[serde(default)]
    pub sidecar_files: Vec<SidecarFile>,
    /// Ordered list of tensors in the archive
    pub tensors: Vec<TensorEntry>,
    /// SHA-256 of the raw tensor data section (everything after the header)
    pub data_sha256: String,
}

/// Per-tensor metadata stored in the header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offset of this tensor's data within the data section
    pub data_offset: u64,
    /// Length of the raw data in bytes
    pub data_len: u64,
    /// xxh3-64 hash of the raw tensor bytes — fast integrity check
    pub xxh3: u64,
}

impl TensorEntry {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Writer — streaming, constant RAM regardless of model size
//
// Design: tensor data is written immediately to a temp file next to the
// output path as each tensor arrives.  On finish() the header is prepended
// by writing: [header] then copying the temp data file.  Peak RAM is one
// tensor at a time, not the whole model.
// ─────────────────────────────────────────────────────────────────────────────

pub struct ArchiveWriter {
    path: std::path::PathBuf,
    model_config: serde_json::Value,
    tokenizer: Option<String>,
    sidecar_files: Vec<SidecarFile>,
    /// Accumulated tensor metadata (no raw bytes kept in RAM)
    entries: Vec<TensorEntry>,
    /// Temp file that receives tensor data as tensors are added
    data_file: BufWriter<File>,
    data_tmp_path: std::path::PathBuf,
    /// Running SHA-256 over streamed tensor data
    hasher: sha2::Sha256,
    data_cursor: u64,
    tensor_count: usize,
}

impl ArchiveWriter {
    pub fn new<P: AsRef<Path>>(path: P, model_config: serde_json::Value) -> Self {
        let path = path.as_ref().to_owned();
        // Temp file lives next to the output file so the final rename/copy is
        // on the same filesystem (fast) and cleaned up automatically on drop.
        let data_tmp_path = path.with_extension("blz.tmp");
        let data_file = BufWriter::new(
            File::create(&data_tmp_path)
                .expect("failed to create temp data file for streaming pack"),
        );
        Self {
            path,
            model_config,
            tokenizer: None,
            sidecar_files: Vec::new(),
            entries: Vec::new(),
            data_file,
            data_tmp_path,
            hasher: sha2::Sha256::new(),
            data_cursor: 0,
            tensor_count: 0,
        }
    }

    /// Embed tokenizer.json as a raw string.
    pub fn set_tokenizer(&mut self, raw_json: String) {
        self.tokenizer = Some(raw_json);
    }

    /// Embed any auxiliary file verbatim (base64-encoded in header).
    pub fn add_sidecar(&mut self, filename: &str, bytes: &[u8]) {
        self.sidecar_files.push(SidecarFile::from_bytes(filename, bytes));
    }

    /// Add a tensor — writes raw bytes immediately to disk, keeps only
    /// metadata in RAM.  This is the key change for large-model support.
    pub fn add_tensor(&mut self, name: &str, dtype: DType, shape: Vec<usize>, raw: Vec<u8>) {
        let xxh3 = xxh3_64(&raw);
        let entry = TensorEntry {
            name: name.to_owned(),
            dtype,
            shape,
            data_offset: self.data_cursor,
            data_len: raw.len() as u64,
            xxh3,
        };
        // Stream bytes to disk immediately — no accumulation in RAM
        self.data_file.write_all(&raw)
            .expect("failed to write tensor data to temp file");
        sha2::Digest::update(&mut self.hasher, &raw);
        self.data_cursor += raw.len() as u64;
        self.entries.push(entry);
        self.tensor_count += 1;
        // raw is dropped here — memory freed immediately
    }

    /// Finalise the archive.  Writes the header then appends the temp data file.
    /// Peak RAM: header JSON only (a few MB at most).
    pub fn finish(mut self) -> Result<WriteStats> {
        use sha2::Digest;

        // Flush and close the temp data file
        self.data_file.flush()?;
        drop(self.data_file);

        let data_sha256 = hex::encode(self.hasher.finalize());

        // ── Build header ──
        let header = ArchiveHeader {
            version: FORMAT_VERSION,
            created_at: chrono_now(),
            model_config: self.model_config,
            tokenizer: self.tokenizer,
            sidecar_files: self.sidecar_files,
            tensors: self.entries,
            data_sha256: data_sha256.clone(),
        };
        let header_json = serde_json::to_vec_pretty(&header)?;
        let header_len = header_json.len() as u64;

        // ── Write final archive: [magic][version][header_len][header][data] ──
        let out_file = File::create(&self.path)
            .with_context(|| format!("creating {}", self.path.display()))?;
        let mut w = BufWriter::new(out_file);

        w.write_u64::<LittleEndian>(MAGIC)?;
        w.write_u32::<LittleEndian>(FORMAT_VERSION)?;
        w.write_u64::<LittleEndian>(header_len)?;
        w.write_all(&header_json)?;

        // Stream temp data into the output file in 8MB chunks — constant RAM
        let mut data_reader = File::open(&self.data_tmp_path)
            .context("opening temp data file")?;
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        loop {
            let n = data_reader.read(&mut buf)?;
            if n == 0 { break; }
            w.write_all(&buf[..n])?;
        }
        w.flush()?;
        drop(data_reader);

        // Clean up temp file
        let _ = std::fs::remove_file(&self.data_tmp_path);

        let total_bytes = 8 + 4 + 8 + header_len + self.data_cursor;
        Ok(WriteStats {
            tensors: self.tensor_count,
            total_bytes,
            data_sha256,
        })
    }
}

pub struct WriteStats {
    pub tensors: usize,
    pub total_bytes: u64,
    pub data_sha256: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Reader
// ─────────────────────────────────────────────────────────────────────────────

pub struct ArchiveReader {
    pub header: ArchiveHeader,
    /// Memory-mapped file — zero-copy access to tensor data
    mmap: Mmap,
    /// Byte offset within mmap where the data section begins
    data_start: usize,
}

impl ArchiveReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("opening {}", path.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cur = std::io::Cursor::new(&mmap[..]);

        let magic = cur.read_u64::<LittleEndian>()?;
        if magic != MAGIC {
            bail!("not a BXP archive (bad magic)");
        }
        let version = cur.read_u32::<LittleEndian>()?;
        if version != FORMAT_VERSION {
            bail!("unsupported archive version {version} (expected {FORMAT_VERSION})");
        }
        let header_len = cur.read_u64::<LittleEndian>()? as usize;
        let header_start = cur.position() as usize;
        let header: ArchiveHeader =
            serde_json::from_slice(&mmap[header_start..header_start + header_len])?;
        let data_start = header_start + header_len;

        Ok(Self { header, mmap, data_start })
    }

    /// Return the raw bytes of a named tensor (zero-copy slice into mmap).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .header
            .tensors
            .iter()
            .find(|e| e.name == name)
            .with_context(|| format!("tensor '{name}' not found"))?;
        let start = self.data_start + entry.data_offset as usize;
        let end = start + entry.data_len as usize;
        Ok(&self.mmap[start..end])
    }

    /// Verify every tensor's xxh3 checksum.
    pub fn verify(&self) -> VerifyReport {
        let mut passed = 0usize;
        let mut failed: Vec<String> = Vec::new();
        for entry in &self.header.tensors {
            let start = self.data_start + entry.data_offset as usize;
            let end = start + entry.data_len as usize;
            let actual = xxh3_64(&self.mmap[start..end]);
            if actual == entry.xxh3 {
                passed += 1;
            } else {
                failed.push(entry.name.clone());
            }
        }
        // Also verify the whole-data SHA-256
        let data_end = self.data_start
            + self.header.tensors.last().map_or(0, |e| {
                (e.data_offset + e.data_len) as usize
            });
        let actual_sha = hex::encode(Sha256::digest(&self.mmap[self.data_start..data_end]));
        let sha_ok = actual_sha == self.header.data_sha256;

        VerifyReport { passed, failed, sha256_ok: sha_ok, expected_sha256: self.header.data_sha256.clone(), actual_sha256: actual_sha }
    }
}

pub struct VerifyReport {
    pub passed: usize,
    pub failed: Vec<String>,
    pub sha256_ok: bool,
    pub expected_sha256: String,
    pub actual_sha256: String,
}

impl VerifyReport {
    pub fn is_clean(&self) -> bool {
        self.failed.is_empty() && self.sha256_ok
    }
}

fn chrono_now() -> String {
    // Simple RFC-3339-ish timestamp without pulling chrono
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}
