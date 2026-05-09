//! Diff and patch for BXP archives
//!
//! `diff`  — compare two archives tensor-by-tensor via xxh3 checksums and
//!            emit a compact binary patch file containing only changed tensors.
//!
//! `apply` — reconstruct the target archive from a base archive + patch file.
//!
//! Patch file layout:
//!   [PATCH_MAGIC 8B][FORMAT_VERSION 4B][MANIFEST_LEN 8B][MANIFEST JSON][BLOB_DATA...]
//!
//! **Format version 2** (current writer): each Modified/Added blob is BLXD-wrapped
//! delta compression (`crate::delta_patch`) when beneficial; version **1** blobs
//! are raw tensor bytes (backward compatible).
//!
//! The manifest lists every tensor with an op tag:
//!   Unchanged — not stored in the patch; taken from base
//!   Modified  — delta- or raw-encoded in patch blob section
//!   Added     — new tensor not in base; encoded in patch blob section
//!   Removed   — tensor present in base but absent in target
//!
//! The new archive header (config, tokenizer, tensor ordering) is stored
//! verbatim inside the manifest so the patch is self-sufficient.

use crate::delta_patch;
use crate::format::{ArchiveReader, ArchiveWriter};
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use xxhash_rust::xxh3::xxh3_64;

pub const PATCH_MAGIC: u64 = 0x0A48435450585A42; // "BZXPTCH\n"

/// On-disk patch container format. **1** = raw tensor blobs; **2** = BLXD delta encoding.
pub const PATCH_FILE_FORMAT_V1: u32 = 1;
pub const PATCH_FILE_FORMAT_V2: u32 = 2;

/// Manifest JSON `version` field for newly written patches.
pub const PATCH_MANIFEST_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DiffOp {
    Unchanged,
    Modified,
    Added,
    Removed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchEntry {
    pub name: String,
    pub op: DiffOp,
    /// Byte offset of this tensor's blob in the patch data section
    /// (only meaningful for Modified / Added)
    pub blob_offset: u64,
    pub blob_len: u64,
    /// xxh3 of the new data (for verification after apply)
    pub xxh3_new: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PatchManifest {
    pub version: u32,
    pub created_at: String,
    /// Full archive header for the target — stored verbatim so apply can
    /// reconstruct the archive without the target being present
    pub target_header: crate::format::ArchiveHeader,
    pub entries: Vec<PatchEntry>,
}

#[derive(Debug, Default)]
pub struct DiffStats {
    pub unchanged: usize,
    pub modified: usize,
    pub added: usize,
    pub removed: usize,
    pub patch_bytes: u64,
    pub base_data_bytes: u64,
    pub patch_data_bytes: u64,
}

impl DiffStats {
    pub fn reduction_pct(&self) -> f64 {
        if self.base_data_bytes == 0 {
            return 0.0;
        }
        100.0 * (1.0 - self.patch_data_bytes as f64 / self.base_data_bytes as f64)
    }
}

/// Create a patch file that transforms `base` into `target`.
pub fn diff<P: AsRef<Path>>(base_path: P, target_path: P, patch_path: P) -> Result<DiffStats> {
    let base = ArchiveReader::open(&base_path)
        .with_context(|| format!("opening base {}", base_path.as_ref().display()))?;
    let target = ArchiveReader::open(&target_path)
        .with_context(|| format!("opening target {}", target_path.as_ref().display()))?;

    // Index base tensors by name → xxh3
    let base_index: HashMap<String, u64> = base
        .header
        .tensors
        .iter()
        .map(|e| (e.name.clone(), e.xxh3))
        .collect();

    let target_names: std::collections::HashSet<String> =
        target.header.tensors.iter().map(|e| e.name.clone()).collect();

    let pb = ProgressBar::new(target.header.tensors.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut entries: Vec<PatchEntry> = Vec::new();
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut blob_cursor: u64 = 0;
    let mut stats = DiffStats::default();

    // Process tensors present in target
    for te in &target.header.tensors {
        pb.set_message(te.name.clone());
        let new_bytes = target.tensor_bytes(&te.name)?;
        stats.base_data_bytes += new_bytes.len() as u64;

        let op = match base_index.get(&te.name) {
            Some(&base_xxh3) if base_xxh3 == te.xxh3 => {
                stats.unchanged += 1;
                DiffOp::Unchanged
            }
            Some(_) => {
                stats.modified += 1;
                DiffOp::Modified
            }
            None => {
                stats.added += 1;
                DiffOp::Added
            }
        };

        let (blob_offset, blob_len) = if op == DiffOp::Unchanged {
            (0, 0)
        } else {
            let off = blob_cursor;
            let blob = match op {
                DiffOp::Modified => {
                    let base_bytes = base.tensor_bytes(&te.name)?;
                    delta_patch::encode_modified(base_bytes, new_bytes, te.dtype)?
                }
                DiffOp::Added => delta_patch::encode_added(new_bytes)?,
                _ => unreachable!(),
            };
            let len = blob.len() as u64;
            blobs.push(blob);
            blob_cursor += len;
            stats.patch_data_bytes += len;
            (off, len)
        };

        entries.push(PatchEntry {
            name: te.name.clone(),
            op,
            blob_offset,
            blob_len,
            xxh3_new: te.xxh3,
        });
        pb.inc(1);
    }

    // Removed tensors (in base but not in target)
    for be in &base.header.tensors {
        if !target_names.contains(&be.name) {
            stats.removed += 1;
            entries.push(PatchEntry {
                name: be.name.clone(),
                op: DiffOp::Removed,
                blob_offset: 0,
                blob_len: 0,
                xxh3_new: 0,
            });
        }
    }
    pb.finish_and_clear();

    let manifest = PatchManifest {
        version: PATCH_MANIFEST_VERSION,
        created_at: timestamp_now(),
        target_header: target.header.clone(),
        entries,
    };

    // Write patch file
    let manifest_json = serde_json::to_vec_pretty(&manifest)?;
    let manifest_len = manifest_json.len() as u64;

    let f = File::create(patch_path.as_ref())
        .with_context(|| format!("creating {}", patch_path.as_ref().display()))?;
    let mut w = BufWriter::new(f);
    w.write_u64::<LittleEndian>(PATCH_MAGIC)?;
    w.write_u32::<LittleEndian>(PATCH_FILE_FORMAT_V2)?;
    w.write_u64::<LittleEndian>(manifest_len)?;
    w.write_all(&manifest_json)?;
    for blob in &blobs {
        w.write_all(blob)?;
    }
    w.flush()?;

    stats.patch_bytes = 8 + 4 + 8 + manifest_len + blob_cursor;
    Ok(stats)
}

/// Apply a patch file to a base archive, writing the result to `out_path`.
pub fn apply<P: AsRef<Path>>(base_path: P, patch_path: P, out_path: P) -> Result<ApplyStats> {
    let base = ArchiveReader::open(&base_path)
        .with_context(|| format!("opening base {}", base_path.as_ref().display()))?;

    // Read patch file
    let patch_bytes = std::fs::read(patch_path.as_ref())
        .with_context(|| format!("reading patch {}", patch_path.as_ref().display()))?;
    let mut cur = std::io::Cursor::new(&patch_bytes[..]);

    let magic = cur.read_u64::<LittleEndian>()?;
    if magic != PATCH_MAGIC {
        anyhow::bail!("not a BXP patch file (bad magic)");
    }
    let patch_file_format = cur.read_u32::<LittleEndian>()?;
    let manifest_len = cur.read_u64::<LittleEndian>()? as usize;
    let manifest_start = cur.position() as usize;
    let manifest: PatchManifest =
        serde_json::from_slice(&patch_bytes[manifest_start..manifest_start + manifest_len])?;
    let blob_start = manifest_start + manifest_len;

    // Build output archive
    let mut writer = ArchiveWriter::new(
        out_path.as_ref(),
        manifest.target_header.model_config.clone(),
    );
    if let Some(tok) = manifest.target_header.tokenizer.clone() {
        writer.set_tokenizer(tok);
    }
    for sidecar in &manifest.target_header.sidecar_files {
        if let Ok(bytes) = sidecar.decode() {
            writer.add_sidecar(&sidecar.filename, &bytes);
        }
    }

    let pb = ProgressBar::new(manifest.entries.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut tensors_applied = 0usize;
    let mut tensors_from_base = 0usize;

    // Walk the manifest in the order the target header defines
    for te in &manifest.target_header.tensors {
        let patch_entry =
            manifest.entries.iter().find(|e| e.name == te.name).with_context(|| {
                format!("patch manifest missing entry for '{}'", te.name)
            })?;

        pb.set_message(te.name.clone());

        let raw: Vec<u8> = match patch_entry.op {
            DiffOp::Unchanged => {
                tensors_from_base += 1;
                base.tensor_bytes(&te.name)?.to_vec()
            }
            DiffOp::Modified => {
                tensors_applied += 1;
                let start = blob_start + patch_entry.blob_offset as usize;
                let end = start + patch_entry.blob_len as usize;
                let blob = &patch_bytes[start..end];
                let base_bytes = base.tensor_bytes(&te.name)?;
                let expected_len = te.data_len as usize;
                if patch_file_format == PATCH_FILE_FORMAT_V1 {
                    if blob.len() != expected_len {
                        anyhow::bail!("v1 patch blob size mismatch for '{}'", te.name);
                    }
                    blob.to_vec()
                } else {
                    delta_patch::decode_blob(blob, Some(base_bytes), expected_len, te.dtype)?
                }
            }
            DiffOp::Added => {
                tensors_applied += 1;
                let start = blob_start + patch_entry.blob_offset as usize;
                let end = start + patch_entry.blob_len as usize;
                let blob = &patch_bytes[start..end];
                let expected_len = te.data_len as usize;
                if patch_file_format == PATCH_FILE_FORMAT_V1 {
                    if blob.len() != expected_len {
                        anyhow::bail!("v1 patch blob size mismatch for '{}'", te.name);
                    }
                    blob.to_vec()
                } else {
                    delta_patch::decode_blob(blob, None, expected_len, te.dtype)?
                }
            }
            DiffOp::Removed => {
                pb.inc(1);
                continue;
            }
        };

        let actual_xxh3 = xxh3_64(&raw);
        if actual_xxh3 != patch_entry.xxh3_new {
            anyhow::bail!(
                "xxh3 mismatch for tensor '{}': expected {:016x} got {:016x}",
                te.name,
                patch_entry.xxh3_new,
                actual_xxh3
            );
        }

        writer.add_tensor(&te.name, te.dtype, te.shape.clone(), raw);
        pb.inc(1);
    }
    pb.finish_and_clear();

    let ws = writer.finish()?;
    Ok(ApplyStats {
        tensors_from_base,
        tensors_from_patch: tensors_applied,
        output_bytes: ws.total_bytes,
        data_sha256: ws.data_sha256,
    })
}

#[derive(Debug)]
pub struct ApplyStats {
    pub tensors_from_base: usize,
    pub tensors_from_patch: usize,
    pub output_bytes: u64,
    pub data_sha256: String,
}

fn timestamp_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}")
}
