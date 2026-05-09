//! blazex-verify-patch — end-to-end diff/patch correctness verifier
//!
//! Given an existing `.blz` archive, this tool:
//!
//!   1. Loads every tensor from the archive
//!   2. Applies configurable mutations to a subset of tensors:
//!        - corrupt:  flip bits in a fraction of values
//!        - scale:    multiply by a constant (simulates fine-tune weight drift)
//!        - zero:     zero out the tensor entirely
//!        - add:      add a new tensor not in the original
//!        - remove:   drop a tensor from the mutated copy
//!   3. Writes the mutated archive as a "target"
//!   4. Runs `diff` to produce a `.blzdiff` patch file
//!   5. Runs `apply` to reconstruct the target from base + patch
//!   6. Performs bit-level verification that every byte of the reconstructed
//!      archive's data section matches the target exactly
//!   7. Verifies that unchanged tensors are genuinely identical to the base
//!   8. Reports a full breakdown: tensors checked, mismatches, patch efficiency
//!
//! Exit code 0 = all checks passed.  Non-zero = something is wrong.
//!
//! Usage:
//!   blazex-verify-patch <archive.blz> [OPTIONS]
//!
//! Options:
//!   --mutate-fraction <f>   Fraction of tensors to mutate  [default: 0.15]
//!   --mutation <type>       corrupt | scale | zero          [default: corrupt]
//!   --add-tensors <n>       Also add N synthetic new tensors [default: 2]
//!   --remove-tensors <n>    Also remove N tensors from target [default: 1]
//!   --seed <u64>            RNG seed for reproducible runs  [default: 42]
//!   --keep-tmp              Don't delete temp files after run
//!   --verbose               Print per-tensor results

use anyhow::{bail, Context, Result};
use blaze_x_pack::{
    format::{ArchiveReader, ArchiveWriter},
    patch,
    types::DType,
};
use std::collections::HashSet;
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    archive: PathBuf,
    mutate_fraction: f64,
    mutation: MutationKind,
    add_tensors: usize,
    remove_tensors: usize,
    seed: u64,
    keep_tmp: bool,
    verbose: bool,
}

#[derive(Debug, Clone, Copy)]
enum MutationKind {
    Corrupt,
    Scale,
    Zero,
}

impl MutationKind {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "corrupt" => Ok(Self::Corrupt),
            "scale"   => Ok(Self::Scale),
            "zero"    => Ok(Self::Zero),
            other     => bail!("Unknown mutation '{other}'. Use: corrupt, scale, zero"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Corrupt => "corrupt",
            Self::Scale   => "scale",
            Self::Zero    => "zero",
        }
    }
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let archive = PathBuf::from(
        args.next().context("Usage: blazex-verify-patch <archive.blz> [OPTIONS]")?
    );
    if !archive.exists() {
        bail!("Archive not found: {}", archive.display());
    }

    let mut mutate_fraction = 0.15f64;
    let mut mutation = MutationKind::Corrupt;
    let mut add_tensors = 2usize;
    let mut remove_tensors = 1usize;
    let mut seed = 42u64;
    let mut keep_tmp = false;
    let mut verbose = false;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--mutate-fraction" => {
                let v = args.next().context("--mutate-fraction needs a value")?;
                mutate_fraction = v.parse().context("--mutate-fraction: expected float")?;
            }
            "--mutation" => {
                let v = args.next().context("--mutation needs a value")?;
                mutation = MutationKind::from_str(&v)?;
            }
            "--add-tensors" => {
                let v = args.next().context("--add-tensors needs a value")?;
                add_tensors = v.parse().context("--add-tensors: expected integer")?;
            }
            "--remove-tensors" => {
                let v = args.next().context("--remove-tensors needs a value")?;
                remove_tensors = v.parse().context("--remove-tensors: expected integer")?;
            }
            "--seed" => {
                let v = args.next().context("--seed needs a value")?;
                seed = v.parse().context("--seed: expected u64")?;
            }
            "--keep-tmp" => keep_tmp = true,
            "--verbose"  => verbose = true,
            other => bail!("Unknown flag '{other}'"),
        }
    }

    Ok(Args { archive, mutate_fraction, mutation, add_tensors, remove_tensors, seed, keep_tmp, verbose })
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal deterministic PRNG — xorshift64, no deps
// ─────────────────────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed ^ 0xdeadbeefcafe1234) }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mutation kernels
// ─────────────────────────────────────────────────────────────────────────────

fn mutate_bytes(raw: &[u8], kind: MutationKind, rng: &mut Rng) -> Vec<u8> {
    let mut out = raw.to_vec();
    match kind {
        MutationKind::Corrupt => {
            // Flip ~5% of bytes at random positions
            let n_flips = ((raw.len() as f64) * 0.05).max(1.0) as usize;
            for _ in 0..n_flips {
                let pos = rng.next_usize(out.len());
                out[pos] ^= 0xFF;
            }
        }
        MutationKind::Scale => {
            // Interpret as f32 slice and multiply by 1.1
            // Works on the raw bytes level without requiring element-count alignment
            for chunk in out.chunks_exact_mut(4) {
                let v = f32::from_le_bytes(chunk.try_into().unwrap());
                let scaled = (v * 1.1).to_le_bytes();
                chunk.copy_from_slice(&scaled);
            }
        }
        MutationKind::Zero => {
            out.fill(0);
        }
    }
    out
}

/// Build a synthetic tensor of given size (F32, values 0..n)
fn synthetic_tensor(name: &str, n_elements: usize, rng: &mut Rng) -> (String, Vec<u8>) {
    let raw: Vec<u8> = (0..n_elements)
        .flat_map(|_| (rng.next_f64() as f32).to_le_bytes())
        .collect();
    (name.to_owned(), raw)
}

// ─────────────────────────────────────────────────────────────────────────────
// Verification
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct VerifyReport {
    tensors_checked: usize,
    byte_mismatches: usize,
    /// Names of tensors where data didn't match
    failed_tensors: Vec<String>,
    /// Tensors that should be unchanged but weren't identical to base
    base_drift: Vec<String>,
    total_bytes_verified: u64,
}

impl VerifyReport {
    fn ok(&self) -> bool {
        self.byte_mismatches == 0 && self.base_drift.is_empty()
    }
}

fn verify_bit_level(
    target: &ArchiveReader,
    reconstructed: &ArchiveReader,
    base: &ArchiveReader,
    mutated_names: &HashSet<String>,
    added_names: &HashSet<String>,
    removed_names: &HashSet<String>,
    verbose: bool,
) -> Result<VerifyReport> {
    let mut report = VerifyReport::default();

    // ── 1. Every tensor in target must appear in reconstructed with identical bytes ──
    for te in &target.header.tensors {
        let target_bytes = target.tensor_bytes(&te.name)?;
        let recon_bytes = match reconstructed.tensor_bytes(&te.name) {
            Ok(b) => b,
            Err(_) => {
                report.failed_tensors.push(format!("{} (missing in reconstructed)", te.name));
                report.byte_mismatches += target_bytes.len();
                report.tensors_checked += 1;
                if verbose {
                    println!("  FAIL  {} — missing in reconstructed archive", te.name);
                }
                continue;
            }
        };

        report.tensors_checked += 1;
        report.total_bytes_verified += target_bytes.len() as u64;

        if target_bytes != recon_bytes {
            // Find first differing byte for diagnostics
            let first_diff = target_bytes.iter().zip(recon_bytes.iter())
                .position(|(a, b)| a != b);
            let n_diff = target_bytes.iter().zip(recon_bytes.iter())
                .filter(|(a, b)| a != b).count();

            report.byte_mismatches += n_diff;
            report.failed_tensors.push(te.name.clone());

            if verbose {
                println!("  FAIL  {} — {} byte(s) differ, first at offset {:?}",
                    te.name, n_diff, first_diff);
            }
        } else if verbose {
            let tag = if mutated_names.contains(&te.name) { "mutated" }
                      else if added_names.contains(&te.name) { "added" }
                      else { "unchanged" };
            println!("  ok    {} ({}, {} bytes)", te.name, tag, target_bytes.len());
        }
    }

    // ── 2. Reconstructed must not contain tensors that were removed ──
    for name in removed_names {
        if reconstructed.tensor_bytes(name).is_ok() {
            report.failed_tensors.push(format!("{name} (should be removed but present)"));
            report.byte_mismatches += 1;
            if verbose {
                println!("  FAIL  {} — should have been removed but is present", name);
            }
        } else if verbose {
            println!("  ok    {} (correctly absent — removed tensor)", name);
        }
    }

    // ── 3. Unchanged tensors must be byte-for-byte identical to the original base ──
    for te in &base.header.tensors {
        if mutated_names.contains(&te.name)
            || added_names.contains(&te.name)
            || removed_names.contains(&te.name)
        {
            continue; // expected to differ
        }
        let base_bytes = base.tensor_bytes(&te.name)?;
        let recon_bytes = match reconstructed.tensor_bytes(&te.name) {
            Ok(b) => b,
            Err(_) => continue, // already caught above
        };
        if base_bytes != recon_bytes {
            report.base_drift.push(te.name.clone());
            if verbose {
                println!("  WARN  {} — unchanged tensor drifted from base!", te.name);
            }
        }
    }

    Ok(report)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn run() -> Result<()> {
    let args = parse_args()?;
    let mut rng = Rng::new(args.seed);

    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║     BLAZE-X diff/patch correctness verifier     ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("Archive  : {}", args.archive.display());
    println!("Mutation : {} (fraction {:.0}%)", args.mutation.name(), args.mutate_fraction * 100.0);
    println!("Add      : {} synthetic tensor(s)", args.add_tensors);
    println!("Remove   : {} tensor(s)", args.remove_tensors);
    println!("Seed     : {}", args.seed);
    println!();

    // ── Step 1: Load the base archive ──
    println!("── Step 1/6  Loading base archive ──────────────────");
    let base = ArchiveReader::open(&args.archive)
        .context("opening base archive")?;
    let n_tensors = base.header.tensors.len();
    println!("  {} tensors  ({:.2} GB data)",
        n_tensors,
        base.header.tensors.iter().map(|t| t.data_len).sum::<u64>() as f64 / 1e9);

    if n_tensors == 0 {
        bail!("Archive has no tensors — nothing to test");
    }

    // ── Step 2: Build mutated target archive ──
    println!();
    println!("── Step 2/6  Building mutated target archive ────────");

    // Pick tensors to mutate
    let n_to_mutate = ((n_tensors as f64 * args.mutate_fraction).ceil() as usize)
        .min(n_tensors);
    let n_to_remove = args.remove_tensors.min(n_tensors - n_to_mutate);

    let mut indices: Vec<usize> = (0..n_tensors).collect();
    rng.shuffle(&mut indices);
    let mutate_indices: HashSet<usize> = indices[..n_to_mutate].iter().copied().collect();
    let remove_indices: HashSet<usize> = indices[n_to_mutate..n_to_mutate + n_to_remove]
        .iter().copied().collect();

    let mut mutated_names: HashSet<String> = HashSet::new();
    let mut removed_names: HashSet<String> = HashSet::new();
    let mut added_names: HashSet<String> = HashSet::new();

    // Create temp directory
    let tmp_dir = std::env::temp_dir().join(format!("blazex-verify-{}", rng.next_u64()));
    std::fs::create_dir_all(&tmp_dir)?;

    let target_path      = tmp_dir.join("target.blz");
    let patch_path       = tmp_dir.join("delta.blzdiff");
    let reconstructed_path = tmp_dir.join("reconstructed.blz");

    // Build target writer
    let mut writer = ArchiveWriter::new(
        &target_path,
        base.header.model_config.clone(),
    );
    if let Some(tok) = &base.header.tokenizer {
        writer.set_tokenizer(tok.clone());
    }

    let mut n_mutated = 0usize;
    let mut n_removed = 0usize;

    for (i, te) in base.header.tensors.iter().enumerate() {
        if remove_indices.contains(&i) {
            removed_names.insert(te.name.clone());
            n_removed += 1;
            println!("  remove  {}", te.name);
            continue;
        }

        let raw = base.tensor_bytes(&te.name)?.to_vec();

        if mutate_indices.contains(&i) {
            let mutated = mutate_bytes(&raw, args.mutation, &mut rng);
            writer.add_tensor(&te.name, te.dtype, te.shape.clone(), mutated);
            mutated_names.insert(te.name.clone());
            n_mutated += 1;
            println!("  mutate  {} ({} bytes, {})", te.name, te.data_len, args.mutation.name());
        } else {
            writer.add_tensor(&te.name, te.dtype, te.shape.clone(), raw);
        }
    }

    // Add synthetic tensors
    for i in 0..args.add_tensors {
        let name = format!("_synthetic_added_{i}");
        let (_, raw) = synthetic_tensor(&name, 128, &mut rng);
        writer.add_tensor(&name, DType::F32, vec![128], raw);
        added_names.insert(name.clone());
        println!("  add     {} (128 × F32)", name);
    }

    writer.finish().context("writing target archive")?;

    println!();
    println!("  {} tensor(s) mutated, {} removed, {} added",
        n_mutated, n_removed, args.add_tensors);

    // ── Step 3: Diff ──
    println!();
    println!("── Step 3/6  Creating patch (diff) ─────────────────");
    let diff_stats = patch::diff(&args.archive, &target_path, &patch_path)
        .context("diff failed")?;

    let patch_size = std::fs::metadata(&patch_path)?.len();
    println!("  Unchanged : {}", diff_stats.unchanged);
    println!("  Modified  : {}", diff_stats.modified);
    println!("  Added     : {}", diff_stats.added);
    println!("  Removed   : {}", diff_stats.removed);
    println!("  Patch size: {:.2} MB  ({:.1}% of base data)",
        patch_size as f64 / 1e6,
        diff_stats.patch_data_bytes as f64 / diff_stats.base_data_bytes.max(1) as f64 * 100.0);

    // Sanity: unchanged count should equal (n_tensors - n_mutated - n_removed)
    let expected_unchanged = n_tensors - n_mutated - n_removed;
    if diff_stats.unchanged != expected_unchanged {
        println!("  WARNING: expected {} unchanged tensors but diff reports {}",
            expected_unchanged, diff_stats.unchanged);
    }

    // ── Step 4: Apply patch ──
    println!();
    println!("── Step 4/6  Applying patch ─────────────────────────");
    let apply_stats = patch::apply(&args.archive, &patch_path, &reconstructed_path)
        .context("apply failed")?;

    println!("  Tensors from base  : {}", apply_stats.tensors_from_base);
    println!("  Tensors from patch : {}", apply_stats.tensors_from_patch);
    println!("  Output size        : {:.2} MB", apply_stats.output_bytes as f64 / 1e6);

    // ── Step 5: Bit-level verification ──
    println!();
    println!("── Step 5/6  Bit-level verification ─────────────────");
    if args.verbose {
        println!();
    }

    let target_reader       = ArchiveReader::open(&target_path)?;
    let reconstructed_reader = ArchiveReader::open(&reconstructed_path)?;

    let report = verify_bit_level(
        &target_reader,
        &reconstructed_reader,
        &base,
        &mutated_names,
        &added_names,
        &removed_names,
        args.verbose,
    )?;

    if args.verbose { println!(); }

    println!("  Tensors checked    : {}", report.tensors_checked);
    println!("  Bytes verified     : {:.2} MB", report.total_bytes_verified as f64 / 1e6);

    // ── Step 6: Checksum cross-check ──
    println!();
    println!("── Step 6/6  Archive-level checksum cross-check ─────");

    // Verify both archives independently with their own SHA-256
    let target_verify       = target_reader.verify();
    let recon_verify        = reconstructed_reader.verify();

    let target_sha_ok = target_verify.sha256_ok;
    let recon_sha_ok  = recon_verify.sha256_ok;
    let target_tensors_ok = target_verify.failed.is_empty();
    let recon_tensors_ok  = recon_verify.failed.is_empty();

    println!("  Target archive     : SHA-256 {} | tensors {}",
        if target_sha_ok   { "OK" } else { "FAIL" },
        if target_tensors_ok { "OK" } else { "FAIL" });
    println!("  Reconstructed      : SHA-256 {} | tensors {}",
        if recon_sha_ok    { "OK" } else { "FAIL" },
        if recon_tensors_ok  { "OK" } else { "FAIL" });

    // ── Results ──
    println!();
    println!("══════════════════════════════════════════════════════");

    let all_ok = report.ok()
        && target_sha_ok && recon_sha_ok
        && target_tensors_ok && recon_tensors_ok;

    if all_ok {
        println!("  ✓  ALL CHECKS PASSED");
        println!();
        println!("  {} tensor(s) verified bit-for-bit across diff/patch cycle",
            report.tensors_checked);
        println!("  {:.2} MB of tensor data confirmed correct",
            report.total_bytes_verified as f64 / 1e6);
        if n_mutated > 0 {
            println!("  {} tensor(s) correctly show mutation", n_mutated);
        }
        if n_removed > 0 {
            println!("  {} tensor(s) correctly absent after removal", n_removed);
        }
        if args.add_tensors > 0 {
            println!("  {} synthetic tensor(s) correctly round-tripped", args.add_tensors);
        }
    } else {
        println!("  ✗  VERIFICATION FAILED");
        println!();
        if !report.failed_tensors.is_empty() {
            println!("  Tensor data mismatches ({}):", report.failed_tensors.len());
            for name in &report.failed_tensors {
                println!("    - {name}");
            }
        }
        if !report.base_drift.is_empty() {
            println!("  Unchanged tensors drifted from base ({}):", report.base_drift.len());
            for name in &report.base_drift {
                println!("    - {name}");
            }
        }
        if report.byte_mismatches > 0 {
            println!("  Total byte mismatches: {}", report.byte_mismatches);
        }
        if !target_sha_ok { println!("  Target archive SHA-256 invalid"); }
        if !recon_sha_ok  { println!("  Reconstructed archive SHA-256 invalid"); }
    }

    println!("══════════════════════════════════════════════════════");
    println!();

    // ── Cleanup ──
    if !args.keep_tmp {
        std::fs::remove_dir_all(&tmp_dir).ok();
    } else {
        println!("Temp files kept at: {}", tmp_dir.display());
        println!("  target:        {}", target_path.display());
        println!("  patch:         {}", patch_path.display());
        println!("  reconstructed: {}", reconstructed_path.display());
        println!();
    }

    if all_ok { Ok(()) } else { bail!("Verification failed") }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
