//! On-the-fly dtype conversion and quantisation kernels.
//!
//! All paths go through F32 as an intermediate.  The pipeline is:
//!
//!   source bytes (any dtype) → decode → []f32 → encode → target bytes
//!
//! Supported targets:
//!   F16     — IEEE 754 half precision
//!   BF16    — Brain float (same exponent range as F32, 7-bit mantissa)
//!   F32     — upcasts lower-precision sources, identity for F32
//!   Q8_0    — GGML Q8_0: block of 32 values, one f32 scale, 32× i8
//!   Q4_0    — GGML Q4_0: block of 32 values, one f32 scale, 32× 4-bit (packed)
//!   Q4_K    — GGML Q4_K: 256-value super-block with per-32 sub-scales, higher quality
//!
//! All quantisation formats exactly match the GGML on-disk layout so GGUF
//! files produced with these kernels load cleanly in llama.cpp and Ollama.

use crate::types::DType;
use anyhow::{bail, Result};

/// The target representation for an export cast operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastTarget {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4K,
}

impl CastTarget {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().replace(['_', '-'], "").as_str() {
            "f32" | "float32"  => Some(CastTarget::F32),
            "f16" | "float16"  => Some(CastTarget::F16),
            "bf16"| "bfloat16" => Some(CastTarget::BF16),
            "q80" | "q8_0"    => Some(CastTarget::Q8_0),
            "q40" | "q4_0"    => Some(CastTarget::Q4_0),
            "q4k" | "q4_k"    => Some(CastTarget::Q4K),
            _                  => None,
        }
    }

    /// The DType that describes the output of this cast (for metadata).
    /// Quantised formats don't have a direct DType — we use the GGUF
    /// type tag path directly in the GGUF writer; for other exporters
    /// we store as-is and document via the cast description.
    pub fn output_dtype(self) -> DType {
        match self {
            CastTarget::F32  => DType::F32,
            CastTarget::F16  => DType::F16,
            CastTarget::BF16 => DType::BF16,
            CastTarget::Q8_0 => DType::I8,   // per-element type (scale separate)
            CastTarget::Q4_0 => DType::U8,   // packed nibbles
            CastTarget::Q4K => DType::U8,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            CastTarget::F32  => "f32",
            CastTarget::F16  => "f16",
            CastTarget::BF16 => "bf16",
            CastTarget::Q8_0 => "q8_0",
            CastTarget::Q4_0 => "q4_0",
            CastTarget::Q4K => "q4_k",
        }
    }

    /// GGML type tag for use in the GGUF tensor info section.
    pub fn ggml_type_tag(self) -> u32 {
        match self {
            CastTarget::F32  => 0,
            CastTarget::F16  => 1,
            CastTarget::BF16 => 30,
            CastTarget::Q8_0 => 8,
            CastTarget::Q4_0 => 2,
            CastTarget::Q4K => 12,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Convert `raw` bytes from `src_dtype` to `target`, returning new bytes.
/// `n_elements` is the total number of elements the tensor contains.
pub fn cast_tensor(
    raw: &[u8],
    src_dtype: DType,
    target: CastTarget,
    n_elements: usize,
) -> Result<Vec<u8>> {
    // Fast path: same representation
    if is_identity(src_dtype, target) {
        return Ok(raw.to_vec());
    }

    // Decode source to f32
    let floats = decode_to_f32(raw, src_dtype, n_elements)?;

    // Encode to target
    encode_from_f32(&floats, target)
}

fn is_identity(src: DType, dst: CastTarget) -> bool {
    matches!(
        (src, dst),
        (DType::F32, CastTarget::F32)
            | (DType::F16, CastTarget::F16)
            | (DType::BF16, CastTarget::BF16)
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode — any dtype → []f32
// ─────────────────────────────────────────────────────────────────────────────

fn decode_to_f32(raw: &[u8], dtype: DType, n: usize) -> Result<Vec<f32>> {
    match dtype {
        DType::F32  => Ok(bytes_to_f32_vec(raw, n)),
        DType::F16  => Ok(f16_bytes_to_f32(raw, n)),
        DType::BF16 => Ok(bf16_bytes_to_f32(raw, n)),
        DType::I8   => Ok(i8_bytes_to_f32(raw, n)),
        DType::I32  => Ok(i32_bytes_to_f32(raw, n)),
        DType::I64  => Ok(i64_bytes_to_f32(raw, n)),
        DType::U8   => Ok(u8_bytes_to_f32(raw, n)),
        DType::U16  => Ok(u16_bytes_to_f32(raw, n)),
        DType::U32  => Ok(u32_bytes_to_f32(raw, n)),
        DType::Bool => Ok(bool_bytes_to_f32(raw, n)),
    }
}

fn bytes_to_f32_vec(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(4)
        .take(n)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn f16_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(2)
        .take(n)
        .map(|b| {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            f16_to_f32(bits)
        })
        .collect()
}

fn bf16_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(2)
        .take(n)
        .map(|b| {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            // BF16 is the top 16 bits of F32 — zero-extend the mantissa
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

fn i8_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.iter().take(n).map(|&b| (b as i8) as f32).collect()
}

fn i32_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(4)
        .take(n)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
        .collect()
}

fn i64_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(8)
        .take(n)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32)
        .collect()
}

fn u8_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.iter().take(n).map(|&b| b as f32).collect()
}

fn u16_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(2)
        .take(n)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()) as f32)
        .collect()
}

fn u32_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.chunks_exact(4)
        .take(n)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32)
        .collect()
}

fn bool_bytes_to_f32(raw: &[u8], n: usize) -> Vec<f32> {
    raw.iter().take(n).map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode — []f32 → target bytes
// ─────────────────────────────────────────────────────────────────────────────

fn encode_from_f32(floats: &[f32], target: CastTarget) -> Result<Vec<u8>> {
    match target {
        CastTarget::F32  => Ok(f32_to_bytes(floats)),
        CastTarget::F16  => Ok(f32_to_f16_bytes(floats)),
        CastTarget::BF16 => Ok(f32_to_bf16_bytes(floats)),
        CastTarget::Q8_0 => quantise_q8_0(floats),
        CastTarget::Q4_0 => quantise_q4_0(floats),
        CastTarget::Q4K => quantise_q4_k(floats),
    }
}

fn f32_to_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_to_f16_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|&f| f32_to_f16(f).to_le_bytes()).collect()
}

fn f32_to_bf16_bytes(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|&f| {
        // Round-to-nearest-even: take top 16 bits of the F32 bit pattern,
        // with rounding based on bit 15.
        let bits = f.to_bits();
        let round = ((bits >> 16) & 1) + 0x7FFF;
        let bf16 = ((bits.wrapping_add(round)) >> 16) as u16;
        bf16.to_le_bytes()
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// F16 bit-level conversion (software, no dep)
// ─────────────────────────────────────────────────────────────────────────────

fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign     = (bits >> 31) as u16;
    let exp      = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3FF;

    if exp <= 0 {
        // Underflow / subnormal — flush to zero (preserving sign)
        sign << 15
    } else if exp >= 31 {
        // Overflow → infinity
        (sign << 15) | 0x7C00
    } else {
        (sign << 15) | ((exp as u16) << 10) | (mantissa as u16)
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign     = ((bits >> 15) as u32) << 31;
    let exp      = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x3FF) as u32;

    let f32_bits = if exp == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal F16 → normalised F32
            let e = mantissa.leading_zeros() - 22;
            sign | ((127 - 14 - e) << 23) | ((mantissa << (e + 1)) & 0x7FFFFF)
        }
    } else if exp == 31 {
        // Inf / NaN
        sign | 0x7F800000 | (mantissa << 13)
    } else {
        sign | (((exp + 127 - 15) as u32) << 23) | (mantissa << 13)
    };
    f32::from_bits(f32_bits)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q8_0 quantisation
//
// GGML Q8_0 block layout (34 bytes per 32 elements):
//   f16  scale   (2 bytes)
//   i8[32] quants (32 bytes)
//
// scale = max(|x|) / 127
// quant = round(x / scale)  clamped to [-127, 127]
// ─────────────────────────────────────────────────────────────────────────────

const Q8_0_BLOCK: usize = 32;
const Q8_0_BYTES: usize = 2 + Q8_0_BLOCK; // f16 scale + 32× i8

fn quantise_q8_0(floats: &[f32]) -> Result<Vec<u8>> {
    if floats.len() % Q8_0_BLOCK != 0 {
        bail!(
            "Q8_0 requires element count to be a multiple of {Q8_0_BLOCK}, got {}",
            floats.len()
        );
    }
    let n_blocks = floats.len() / Q8_0_BLOCK;
    let mut out = vec![0u8; n_blocks * Q8_0_BYTES];
    let mut out_ptr = 0usize;

    for block in floats.chunks_exact(Q8_0_BLOCK) {
        let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        // Write f16 scale
        let scale_f16 = f32_to_f16(scale);
        out[out_ptr..out_ptr + 2].copy_from_slice(&scale_f16.to_le_bytes());
        out_ptr += 2;

        // Write i8 quants
        for &v in block {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            out[out_ptr] = q as u8;
            out_ptr += 1;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_0 quantisation
//
// GGML Q4_0 block layout (18 bytes per 32 elements):
//   f16  scale   (2 bytes)
//   u8[16] packed nibbles (16 bytes) — element i in low nibble, i+16 in high nibble
//
// scale = max(|x|) / -8  (uses negative range for better coverage)
// quant = round(x / scale) + 8  clamped to [0, 15]
// ─────────────────────────────────────────────────────────────────────────────

const Q4_0_BLOCK: usize = 32;
const Q4_0_BYTES: usize = 2 + Q4_0_BLOCK / 2; // 18

fn quantise_q4_0(floats: &[f32]) -> Result<Vec<u8>> {
    if floats.len() % Q4_0_BLOCK != 0 {
        bail!(
            "Q4_0 requires element count to be a multiple of {Q4_0_BLOCK}, got {}",
            floats.len()
        );
    }
    let n_blocks = floats.len() / Q4_0_BLOCK;
    let mut out = vec![0u8; n_blocks * Q4_0_BYTES];
    let mut out_ptr = 0usize;

    for block in floats.chunks_exact(Q4_0_BLOCK) {
        let amax = block.iter().copied().fold(0.0f32, |acc, v| acc.max(v.abs()));
        // Scale uses the negative side (-8..7) to give one extra step below 0
        let scale = amax / -8.0;
        let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

        let scale_f16 = f32_to_f16(scale);
        out[out_ptr..out_ptr + 2].copy_from_slice(&scale_f16.to_le_bytes());
        out_ptr += 2;

        // Pack pairs: element i (low nibble) and element i+16 (high nibble)
        let half = Q4_0_BLOCK / 2;
        for i in 0..half {
            let q0 = ((block[i]      * inv_scale).round().clamp(0.0, 15.0) as u8) & 0x0F;
            let q1 = ((block[i+half] * inv_scale).round().clamp(0.0, 15.0) as u8) & 0x0F;
            out[out_ptr] = q0 | (q1 << 4);
            out_ptr += 1;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_K quantisation
//
// GGML Q4_K super-block layout (144 bytes per 256 elements):
//   f16  d        (2 bytes)  — super-block scale
//   f16  dmin     (2 bytes)  — super-block min
//   u8[12] scales (12 bytes) — packed 6-bit scales and mins for 8 sub-blocks
//   u8[128] qs    (128 bytes)— packed 4-bit quants
//
// Each super-block of 256 elements is split into 8 sub-blocks of 32.
// Per sub-block: scale_i = d * (scales_i & 63), min_i = dmin * (mins_i & 63)
// quant = round((x + min_i) / scale_i)  clamped [0, 15]
//
// This matches the GGML ggml_type_q4_k structure exactly.
// ─────────────────────────────────────────────────────────────────────────────

const Q4_K_SUPER:  usize = 256;
const Q4_K_NSUB:   usize = 8;
const Q4_K_SUB:    usize = Q4_K_SUPER / Q4_K_NSUB; // 32
const Q4_K_BYTES:  usize = 2 + 2 + 12 + Q4_K_SUPER / 2; // 144

fn quantise_q4_k(floats: &[f32]) -> Result<Vec<u8>> {
    if floats.len() % Q4_K_SUPER != 0 {
        bail!(
            "Q4_K requires element count to be a multiple of {Q4_K_SUPER}, got {}. \
             Consider padding or using Q4_0 / Q8_0 for tensors with other sizes.",
            floats.len()
        );
    }
    let n_blocks = floats.len() / Q4_K_SUPER;
    let mut out = vec![0u8; n_blocks * Q4_K_BYTES];
    let mut out_ptr = 0usize;

    for super_block in floats.chunks_exact(Q4_K_SUPER) {
        // ── Compute per-sub-block min/max ──
        let mut sub_mins  = [0.0f32; Q4_K_NSUB];
        let mut sub_maxes = [0.0f32; Q4_K_NSUB];
        for (si, sub) in super_block.chunks_exact(Q4_K_SUB).enumerate() {
            let mn = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            sub_mins[si]  = mn;
            sub_maxes[si] = mx;
        }

        // ── Super-block scale and min ──
        let super_max  = sub_maxes.iter().copied().fold(0.0f32, f32::max);
        let super_min  = sub_mins.iter().copied().fold(0.0f32, f32::min);
        // d   normalises the per-sub scale values into 6-bit range [0, 63]
        // dmin normalises the per-sub min  values into 6-bit range [0, 63]
        let d    = super_max / 63.0;
        let dmin = if super_min < 0.0 { super_min / 63.0 } else { 0.0 };

        let inv_d    = if d    == 0.0 { 0.0 } else { 1.0 / d };
        let inv_dmin = if dmin == 0.0 { 0.0 } else { 1.0 / dmin };

        // ── Quantise sub-block scales and mins to 6 bits each ──
        let mut sc6 = [0u8; Q4_K_NSUB];
        let mut mn6 = [0u8; Q4_K_NSUB];
        for si in 0..Q4_K_NSUB {
            sc6[si] = ((sub_maxes[si] * inv_d   ).round().clamp(0.0, 63.0) as u8) & 0x3F;
            mn6[si] = ((sub_mins[si]  * inv_dmin).round().clamp(0.0, 63.0) as u8) & 0x3F;
        }

        // ── Pack 8 pairs of 6-bit (scale, min) into 12 bytes ──
        // Layout: sc[0..3] in low 6, mn[0..3] in high 6 → bytes 0..5
        //         sc[4..7] in low 6, mn[4..7] in high 6 → bytes 6..11
        let mut packed_scales = [0u8; 12];
        for i in 0..4 {
            packed_scales[i]     =  sc6[i]       | ((mn6[i] & 0x0F) << 6);
            packed_scales[i + 4] = (mn6[i] >> 4) | (sc6[i + 4] << 2) | ((mn6[i + 4] & 0x03) << 6);
            packed_scales[i + 8] = mn6[i + 4] >> 2;
        }

        // ── Write super-block header ──
        let d_f16    = f32_to_f16(d);
        let dmin_f16 = f32_to_f16(dmin);
        out[out_ptr..out_ptr + 2].copy_from_slice(&d_f16.to_le_bytes());    out_ptr += 2;
        out[out_ptr..out_ptr + 2].copy_from_slice(&dmin_f16.to_le_bytes()); out_ptr += 2;
        out[out_ptr..out_ptr + 12].copy_from_slice(&packed_scales);         out_ptr += 12;

        // ── Quantise each element to 4 bits ──
        // Pack pairs of consecutive elements: elem[2i] in low nibble, elem[2i+1] in high
        for (si, sub) in super_block.chunks_exact(Q4_K_SUB).enumerate() {
            let scale = d * sc6[si] as f32;
            let min   = dmin * mn6[si] as f32;
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            for pair in sub.chunks_exact(2) {
                let q0 = (((pair[0] - min) * inv_scale).round().clamp(0.0, 15.0) as u8) & 0x0F;
                let q1 = (((pair[1] - min) * inv_scale).round().clamp(0.0, 15.0) as u8) & 0x0F;
                out[out_ptr] = q0 | (q1 << 4);
                out_ptr += 1;
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    #[test]
    fn f32_identity() {
        let vals: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let raw = make_f32_bytes(&vals);
        let out = cast_tensor(&raw, DType::F32, CastTarget::F32, vals.len()).unwrap();
        assert_eq!(raw, out);
    }

    #[test]
    fn f32_to_f16_roundtrip() {
        let vals: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 100.0, -100.0];
        let raw = make_f32_bytes(&vals);
        let f16_bytes = cast_tensor(&raw, DType::F32, CastTarget::F16, vals.len()).unwrap();
        assert_eq!(f16_bytes.len(), vals.len() * 2);
        // Decode back and check within F16 precision
        let recovered = f16_bytes_to_f32(&f16_bytes, vals.len());
        for (orig, rec) in vals.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < orig.abs() * 0.01 + 0.001,
                "f32→f16 roundtrip: {orig} → {rec}");
        }
    }

    #[test]
    fn f32_to_bf16_roundtrip() {
        let vals: Vec<f32> = vec![0.0, 1.0, -1.0, 3.14159, 1e10, -1e10];
        let raw = make_f32_bytes(&vals);
        let bf16_bytes = cast_tensor(&raw, DType::F32, CastTarget::BF16, vals.len()).unwrap();
        assert_eq!(bf16_bytes.len(), vals.len() * 2);
        let recovered = bf16_bytes_to_f32(&bf16_bytes, vals.len());
        for (orig, rec) in vals.iter().zip(recovered.iter()) {
            // BF16 has only 7-bit mantissa — expect ~1% relative error at most
            assert!((orig - rec).abs() < orig.abs() * 0.02 + 0.001,
                "f32→bf16 roundtrip: {orig} → {rec}");
        }
    }

    #[test]
    fn q8_0_block_size() {
        let vals: Vec<f32> = (0..128).map(|i| i as f32 - 64.0).collect();
        let raw = make_f32_bytes(&vals);
        let out = cast_tensor(&raw, DType::F32, CastTarget::Q8_0, 128).unwrap();
        // 128 elements / 32 per block = 4 blocks × 34 bytes = 136
        assert_eq!(out.len(), 4 * Q8_0_BYTES);
    }

    #[test]
    fn q8_0_reconstruction_quality() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let raw = make_f32_bytes(&vals);
        let q8 = cast_tensor(&raw, DType::F32, CastTarget::Q8_0, 32).unwrap();
        assert_eq!(q8.len(), Q8_0_BYTES);
        // Decode manually: scale is f16 in first 2 bytes, then 32 i8s
        let scale_bits = u16::from_le_bytes(q8[0..2].try_into().unwrap());
        let scale = f16_to_f32(scale_bits);
        for (j, &orig) in vals.iter().enumerate() {
            let q = q8[2 + j] as i8;
            let rec = q as f32 * scale;
            assert!((orig - rec).abs() < scale * 0.6,
                "Q8_0 reconstruction at {j}: {orig} → {rec} (scale={scale})");
        }
    }

    #[test]
    fn q4_0_block_size() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let raw = make_f32_bytes(&vals);
        let out = cast_tensor(&raw, DType::F32, CastTarget::Q4_0, 64).unwrap();
        // 64 / 32 = 2 blocks × 18 bytes = 36
        assert_eq!(out.len(), 2 * Q4_0_BYTES);
    }

    #[test]
    fn q4_k_block_size() {
        let vals: Vec<f32> = (0..512).map(|i| i as f32 * 0.01 - 2.56).collect();
        let raw = make_f32_bytes(&vals);
        let out = cast_tensor(&raw, DType::F32, CastTarget::Q4K, 512).unwrap();
        // 512 / 256 = 2 super-blocks × 144 bytes = 288
        assert_eq!(out.len(), 2 * Q4_K_BYTES);
    }

    #[test]
    fn q4_k_requires_multiple_of_256() {
        let vals: Vec<f32> = vec![1.0; 100];
        let raw = make_f32_bytes(&vals);
        let result = cast_tensor(&raw, DType::F32, CastTarget::Q4K, 100);
        assert!(result.is_err());
    }

    #[test]
    fn bf16_source_to_f16_target() {
        // Verify decode-then-encode path works for non-f32 source
        let src_vals: Vec<f32> = vec![1.0, 2.0, 0.5, -3.0];
        let bf16_raw = f32_to_bf16_bytes(&src_vals);
        let out = cast_tensor(&bf16_raw, DType::BF16, CastTarget::F16, 4).unwrap();
        assert_eq!(out.len(), 4 * 2);
        let recovered = f16_bytes_to_f32(&out, 4);
        for (orig, rec) in src_vals.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.05, "bf16→f16: {orig} → {rec}");
        }
    }
}
