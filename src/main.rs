//! blazex — BLAZE-X model packager
//!
//! Commands:
//!   pack     Pack a HuggingFace model directory into a .blz archive
//!   info     Show archive metadata and statistics
//!   list     List tensors with name / dtype / shape / size
//!   extract  Extract one or more tensors to raw .bin files
//!   verify   Verify all tensor checksums and SHA-256 digest
//!   diff     Create a binary patch between two archives
//!   apply    Apply a patch to produce a new archive
//!   export   Export to safetensors / pytorch / gguf / raw

mod cast;
mod codec_ffi;
mod delta_patch;
mod exporter;
mod format;
mod loader;
mod patch;
mod types;

use anyhow::Result;
use clap::{Parser, Subcommand};
use comfy_table::{presets::UTF8_FULL, Cell, Table};
use cast::CastTarget;
use format::ArchiveReader;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "blazex",
    version = "0.1.0",
    author = "BLAZE-X Contributors",
    about = "BXP model packager — pack, diff, patch, verify, export",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pack a HuggingFace model directory into a .blz archive
    Pack {
        /// Source directory (must contain .safetensors files and config.json)
        #[arg(short, long)]
        input: PathBuf,
        /// Output archive path  (e.g. llama3-8b.blz)
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Show archive metadata and statistics
    Info {
        /// Path to .blz archive
        archive: PathBuf,
    },

    /// List tensors in the archive
    List {
        /// Path to .blz archive
        archive: PathBuf,
        /// Filter tensors containing this substring
        #[arg(long)]
        filter: Option<String>,
    },

    /// Extract one or more tensors to .bin files
    Extract {
        /// Path to .blz archive
        #[arg(short, long)]
        archive: PathBuf,
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,
        /// Tensor names to extract (extract all if omitted)
        #[arg(long, num_args = 1..)]
        tensor: Vec<String>,
    },

    /// Verify tensor checksums and data integrity
    Verify {
        /// Path to .blz archive
        archive: PathBuf,
    },

    /// Create a binary patch that transforms base into target
    Diff {
        /// Base archive
        #[arg(long)]
        base: PathBuf,
        /// Target archive
        #[arg(long)]
        target: PathBuf,
        /// Output patch file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Apply a patch file to a base archive
    Apply {
        /// Base archive
        #[arg(long)]
        base: PathBuf,
        /// Patch file produced by `blazex diff`
        #[arg(long)]
        patch: PathBuf,
        /// Output archive path
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Export archive to another format
    Export {
        /// Path to .blz archive
        #[arg(short, long)]
        archive: PathBuf,
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,
        /// Target format: safetensors | pytorch | gguf | raw
        #[arg(long, default_value = "safetensors")]
        to: String,
        /// On-the-fly weight cast: f32 | f16 | bf16 | q8_0 | q4_0 | q4_k
        #[arg(long)]
        cast: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pack { input, output } => cmd_pack(input, output),
        Commands::Info { archive } => cmd_info(archive),
        Commands::List { archive, filter } => cmd_list(archive, filter),
        Commands::Extract { archive, output, tensor } => cmd_extract(archive, output, tensor),
        Commands::Verify { archive } => cmd_verify(archive),
        Commands::Diff { base, target, output } => cmd_diff(base, target, output),
        Commands::Apply { base, patch, output } => cmd_apply(base, patch, output),
        Commands::Export { archive, output, to, cast } => cmd_export(archive, output, to, cast),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// pack
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_pack(input: PathBuf, output: PathBuf) -> Result<()> {
    println!("Packing {} → {}", input.display(), output.display());

    let config = loader::read_config(&input)
        .unwrap_or_else(|_| serde_json::json!({}));
    let tokenizer = loader::read_tokenizer(&input);
    let sidecars = loader::read_sidecar_files(&input);

    let mut writer = format::ArchiveWriter::new(&output, config);
    if let Some(tok) = tokenizer {
        writer.set_tokenizer(tok);
        println!("  Embedded tokenizer.json");
    }
    for (name, bytes) in &sidecars {
        writer.add_sidecar(name, bytes);
        println!("  Embedded {name} ({} bytes)", bytes.len());
    }

    // Stream shards one at a time — each shard is loaded, its tensors written
    // to the archive immediately, then freed.  Peak RAM = one shard (~4-5GB
    // for a 70B model with 30 shards) rather than the full model.
    let shard_paths = loader::shard_paths(&input)?;
    let total_shards = shard_paths.len();
    let mut total_tensors = 0usize;

    for (i, shard) in shard_paths.iter().enumerate() {
        println!("  Packing shard {}/{}: {}", i + 1, total_shards, shard.display());
        let tensors = loader::load_safetensors_file(shard)?;
        let pb = ProgressBar::new(tensors.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("    {spinner:.green} [{bar:35}] {pos}/{len} tensors")
                .unwrap()
                .progress_chars("=> "),
        );
        for t in tensors {
            writer.add_tensor(&t.name, t.dtype, t.shape, t.raw);
            pb.inc(1);
            total_tensors += 1;
        }
        // Shard tensors are dropped here — RAM freed before next shard loads
        pb.finish_and_clear();
    }
    println!("  Packed {} tensors total", total_tensors);

    let stats = writer.finish()?;
    println!(
        "\n✓ Packed {} tensors — {:.2} GB — SHA-256: {}",
        stats.tensors,
        stats.total_bytes as f64 / 1_073_741_824.0,
        &stats.data_sha256[..16]
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// info
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_info(archive: PathBuf) -> Result<()> {
    let r = ArchiveReader::open(&archive)?;
    let h = &r.header;

    let total_bytes: u64 = h.tensors.iter().map(|t| t.data_len).sum();
    let model_type = h.model_config.get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let arch = h.model_config.get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    println!("Archive : {}", archive.display());
    println!("Format  : BXP v{}", h.version);
    println!("Created : {}", h.created_at);
    println!("Model   : {} / {}", model_type, arch);
    println!("Tensors : {}", h.tensors.len());
    println!("Size    : {:.3} GB ({} bytes)", total_bytes as f64 / 1e9, total_bytes);
    println!("SHA-256 : {}", h.data_sha256);
    println!("Tokenizer embedded: {}", if h.tokenizer.is_some() { "yes" } else { "no" });
    if !h.sidecar_files.is_empty() {
        let names: Vec<&str> = h.sidecar_files.iter().map(|s| s.filename.as_str()).collect();
        println!("Sidecar files     : {}", names.join(", "));
    }

    // dtype summary
    let mut dtype_counts: std::collections::HashMap<String, usize> = Default::default();
    for t in &h.tensors {
        *dtype_counts.entry(t.dtype.as_str().to_owned()).or_default() += 1;
    }
    let mut dtypes: Vec<_> = dtype_counts.iter().collect();
    dtypes.sort_by_key(|(k, _)| k.as_str());
    let dtype_str = dtypes.iter().map(|(k, v)| format!("{k}×{v}")).collect::<Vec<_>>().join("  ");
    println!("DTypes  : {}", dtype_str);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// list
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_list(archive: PathBuf, filter: Option<String>) -> Result<()> {
    let r = ArchiveReader::open(&archive)?;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec!["#", "Name", "DType", "Shape", "Size"]);

    for (i, t) in r.header.tensors.iter().enumerate() {
        if let Some(ref f) = filter {
            if !t.name.contains(f.as_str()) {
                continue;
            }
        }
        let shape_str = t.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
        let mb = t.data_len as f64 / 1_048_576.0;
        let size_str = if mb >= 1024.0 {
            format!("{:.2} GB", mb / 1024.0)
        } else {
            format!("{:.1} MB", mb)
        };
        table.add_row(vec![
            Cell::new(i),
            Cell::new(&t.name),
            Cell::new(t.dtype.as_str()),
            Cell::new(shape_str),
            Cell::new(size_str),
        ]);
    }
    println!("{table}");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// extract
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_extract(archive: PathBuf, output: PathBuf, tensors: Vec<String>) -> Result<()> {
    let r = ArchiveReader::open(&archive)?;
    std::fs::create_dir_all(&output)?;

    let to_extract: Vec<&crate::format::TensorEntry> = if tensors.is_empty() {
        r.header.tensors.iter().collect()
    } else {
        r.header.tensors.iter()
            .filter(|t| tensors.contains(&t.name))
            .collect()
    };

    if to_extract.is_empty() {
        println!("No matching tensors found.");
        return Ok(());
    }

    for entry in &to_extract {
        let safe_name = entry.name.replace('/', ".");
        let out_path = output.join(format!("{safe_name}.bin"));
        let raw = r.tensor_bytes(&entry.name)?;
        std::fs::write(&out_path, raw)?;
        let shape_str = entry.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("×");
        println!(
            "  {} → {} ({} {} {:.1} MB)",
            entry.name,
            out_path.display(),
            entry.dtype.as_str(),
            shape_str,
            entry.data_len as f64 / 1_048_576.0
        );
    }
    println!("\n✓ Extracted {} tensor(s) to {}", to_extract.len(), output.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// verify
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_verify(archive: PathBuf) -> Result<()> {
    println!("Verifying {} …", archive.display());
    let r = ArchiveReader::open(&archive)?;

    let pb = ProgressBar::new(r.header.tensors.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40}] {pos}/{len} verifying")
            .unwrap()
            .progress_chars("=> "),
    );
    // Run verification (includes progress internally)
    pb.finish_and_clear();

    let report = r.verify();

    for name in &report.failed {
        println!("  ✗ FAIL: {name}");
    }

    println!("  SHA-256 : {}", if report.sha256_ok { "OK" } else { "FAIL" });
    println!(
        "  Expected: {}",
        &report.expected_sha256[..32]
    );
    if !report.sha256_ok {
        println!("  Actual  : {}", &report.actual_sha256[..32]);
    }
    println!();
    if report.is_clean() {
        println!(
            "✓ {} tensors verified — archive is intact",
            report.passed
        );
    } else {
        println!(
            "✗ VERIFICATION FAILED — {} passed, {} failed",
            report.passed,
            report.failed.len()
        );
        std::process::exit(1);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// diff
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_diff(base: PathBuf, target: PathBuf, output: PathBuf) -> Result<()> {
    println!(
        "Diff\n  base  : {}\n  target: {}\n  patch : {}",
        base.display(),
        target.display(),
        output.display()
    );
    let stats = patch::diff(&base, &target, &output)?;
    println!("\nDiff summary:");
    println!("  Unchanged : {}", stats.unchanged);
    println!("  Modified  : {}", stats.modified);
    println!("  Added     : {}", stats.added);
    println!("  Removed   : {}", stats.removed);
    println!(
        "  Patch size: {:.3} MB ({:.1}% of full model)",
        stats.patch_bytes as f64 / 1e6,
        100.0 - stats.reduction_pct()
    );
    println!("✓ Patch written to {}", output.display());
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// apply
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_apply(base: PathBuf, patch: PathBuf, output: PathBuf) -> Result<()> {
    println!(
        "Apply\n  base : {}\n  patch: {}\n  out  : {}",
        base.display(),
        patch.display(),
        output.display()
    );
    let stats = patch::apply(&base, &patch, &output)?;
    println!(
        "\n✓ Applied: {} from base, {} from patch → {:.3} GB",
        stats.tensors_from_base,
        stats.tensors_from_patch,
        stats.output_bytes as f64 / 1_073_741_824.0
    );
    println!("  SHA-256: {}", &stats.data_sha256[..16]);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// export
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_export(archive: PathBuf, output: PathBuf, to: String, cast_str: Option<String>) -> Result<()> {
    let cast = match cast_str.as_deref() {
        None => None,
        Some(s) => Some(
            CastTarget::from_str(s)
                .ok_or_else(|| anyhow::anyhow!(
                    "Unknown cast target '{s}'. Use: f32, f16, bf16, q8_0, q4_0, q4_k"
                ))?
        ),
    };
    let cast_label = cast.map(|c| format!(" [cast→{}]", c.display_name())).unwrap_or_default();
    println!("Export {} → {} ({}{})", archive.display(), output.display(), to, cast_label);
    let r = ArchiveReader::open(&archive)?;

    let stats = match to.to_lowercase().as_str() {
        "safetensors" | "hf" => exporter::export_safetensors(&r, &output, cast)?,
        "pytorch" | "pt" => exporter::export_pytorch(&r, &output, cast)?,
        "gguf" => {
            // Pass output directly — export_gguf handles both directory paths
            // (writes model.gguf inside) and explicit .gguf file paths.
            exporter::export_gguf(&r, &output, cast)?
        }
        "raw" => exporter::export_raw(&r, &output, cast)?,
        other => anyhow::bail!("Unknown export format '{other}'. Use: safetensors, pytorch, gguf, raw"),
    };

    println!(
        "\n✓ Exported {} tensors ({} file(s), {:.3} GB)",
        stats.tensors,
        stats.files_written,
        stats.bytes_written as f64 / 1_073_741_824.0,
    );
    Ok(())
}
