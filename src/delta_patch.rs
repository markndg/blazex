//! BlazEx delta patch encoding — XOR-based strategies with zstd (SplitStream, SparseDelta, FullXor).
//!
//! Each patch blob (format v2+) starts with `BLXD` + one-byte encoding, then payload.

use crate::codec_ffi;
use crate::types::DType;
use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read};

pub const BLOB_MAGIC: &[u8; 4] = b"BLXD";

/// zstd compression level for patch payloads (balance ratio vs CPU).
pub const ZSTD_LEVEL: i32 = 9;

const TAG_SPARSE_BYTE: u8 = 0x42;
const TAG_SPARSE_U16: u8 = 0x46;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobEncoding {
    Raw = 0,
    SparseByte = 1,
    SparseU16 = 2,
    FullXor = 3,
    SplitStream = 4,
    ZstdOnly = 5,
}

impl BlobEncoding {
    fn from_u8(b: u8) -> Result<Self> {
        match b {
            0 => Ok(BlobEncoding::Raw),
            1 => Ok(BlobEncoding::SparseByte),
            2 => Ok(BlobEncoding::SparseU16),
            3 => Ok(BlobEncoding::FullXor),
            4 => Ok(BlobEncoding::SplitStream),
            5 => Ok(BlobEncoding::ZstdOnly),
            _ => bail!("unknown BLXD encoding tag {b}"),
        }
    }
}

fn compress_zstd(data: &[u8]) -> Result<Vec<u8>> {
    zstd::bulk::compress(data, ZSTD_LEVEL).context("zstd compress")
}

fn decompress_zstd(data: &[u8], max_len: usize) -> Result<Vec<u8>> {
    zstd::bulk::decompress(data, max_len).context("zstd decompress")
}

fn wrap_blxd(encoding: BlobEncoding, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(5 + payload.len());
    v.extend_from_slice(BLOB_MAGIC);
    v.push(encoding as u8);
    v.extend_from_slice(payload);
    v
}

/// Split BLXD wrapper. Returns `(encoding, inner_payload)`.
pub fn unwrap_blxd(blob: &[u8]) -> Result<(BlobEncoding, &[u8])> {
    if blob.len() < 5 {
        bail!("patch blob too short for BLXD header");
    }
    if blob[..4] != *BLOB_MAGIC {
        bail!("patch blob missing BLXD magic");
    }
    let enc = BlobEncoding::from_u8(blob[4])?;
    Ok((enc, &blob[5..]))
}

/// Encode a **modified** tensor (base and target same length).
pub fn encode_modified(base: &[u8], target: &[u8], dtype: DType) -> Result<Vec<u8>> {
    if base.len() != target.len() {
        return Ok(wrap_blxd(BlobEncoding::Raw, target));
    }
    if base.is_empty() {
        return Ok(wrap_blxd(BlobEncoding::Raw, target));
    }

    let elem = dtype.byte_size();
    if base.len() % elem != 0 {
        return Ok(wrap_blxd(BlobEncoding::Raw, target));
    }

    let encoded = if is_float16(dtype) {
        encode_split_stream(base, target)?
    } else if change_rate_elements(base, target, elem) < 0.33 && !is_float(dtype) {
        if elem == 2 {
            encode_sparse_u16(base, target)?
        } else {
            encode_sparse_byte(base, target)?
        }
    } else {
        encode_full_xor(base, target)?
    };

    let raw_wrap = wrap_blxd(BlobEncoding::Raw, target);
    if encoded.len() <= raw_wrap.len() {
        Ok(encoded)
    } else {
        Ok(raw_wrap)
    }
}

/// Encode an **added** tensor (no base): zstd-only or raw inside BLXD.
pub fn encode_added(target: &[u8]) -> Result<Vec<u8>> {
    if target.is_empty() {
        return Ok(wrap_blxd(BlobEncoding::Raw, target));
    }
    let z = compress_zstd(target)?;
    let raw = wrap_blxd(BlobEncoding::Raw, target);
    let z_wrap = wrap_blxd(BlobEncoding::ZstdOnly, &z);
    if z_wrap.len() < raw.len() {
        Ok(z_wrap)
    } else {
        Ok(raw)
    }
}

fn is_float16(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16)
}

fn is_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

fn change_rate_elements(base: &[u8], target: &[u8], elem_size: usize) -> f64 {
    let n = base.len() / elem_size;
    if n == 0 {
        return 0.0;
    }
    let mut changed = 0usize;
    for i in 0..n {
        let a = i * elem_size;
        let b = a + elem_size;
        if base[a..b] != target[a..b] {
            changed += 1;
        }
    }
    changed as f64 / n as f64
}

fn encode_sparse_byte(base: &[u8], target: &[u8]) -> Result<Vec<u8>> {
    let n_bytes = base.len();
    let mut bitmap = vec![0u8; (n_bytes + 7) / 8];
    let mut xor_vals = Vec::new();
    for i in 0..n_bytes {
        if base[i] != target[i] {
            bitmap[i / 8] |= 1 << (i % 8);
            xor_vals.push(base[i] ^ target[i]);
        }
    }
    let n_changed = xor_vals.len() as u32;
    let mut inner = Vec::new();
    inner.push(TAG_SPARSE_BYTE);
    inner.write_u32::<LittleEndian>(n_bytes as u32)?;
    inner.write_u32::<LittleEndian>(n_changed)?;
    inner.extend_from_slice(&bitmap);
    inner.extend_from_slice(&xor_vals);
    let z = compress_zstd(&inner)?;
    Ok(wrap_blxd(BlobEncoding::SparseByte, &z))
}

fn encode_sparse_u16(base: &[u8], target: &[u8]) -> Result<Vec<u8>> {
    let n_values = base.len() / 2;
    let mut bitmap = vec![0u8; (n_values + 7) / 8];
    let mut xor_pairs = Vec::new();
    for i in 0..n_values {
        let o = i * 2;
        let b = u16::from_le_bytes([base[o], base[o + 1]]);
        let t = u16::from_le_bytes([target[o], target[o + 1]]);
        if b != t {
            bitmap[i / 8] |= 1 << (i % 8);
            xor_pairs.extend_from_slice(&(b ^ t).to_le_bytes());
        }
    }
    let n_changed = (xor_pairs.len() / 2) as u32;
    let mut inner = Vec::new();
    inner.push(TAG_SPARSE_U16);
    inner.write_u32::<LittleEndian>(n_values as u32)?;
    inner.write_u32::<LittleEndian>(n_changed)?;
    inner.extend_from_slice(&bitmap);
    inner.extend_from_slice(&xor_pairs);
    let z = compress_zstd(&inner)?;
    Ok(wrap_blxd(BlobEncoding::SparseU16, &z))
}

fn encode_full_xor(base: &[u8], target: &[u8]) -> Result<Vec<u8>> {
    let xor: Vec<u8> = base
        .iter()
        .zip(target.iter())
        .map(|(a, b)| a ^ b)
        .collect();
    let z = compress_zstd(&xor)?;
    Ok(wrap_blxd(BlobEncoding::FullXor, &z))
}

fn encode_split_stream(base: &[u8], target: &[u8]) -> Result<Vec<u8>> {
    let xor: Vec<u8> = base.iter().zip(target.iter()).map(|(a, b)| a ^ b).collect();
    let capacity = xor.len() * 2 + 1024;
    let mut out = vec![0u8; capacity];
    let written = unsafe {
        codec_ffi::blazec_encode_split_stream(
            xor.as_ptr(),
            xor.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
    if written == usize::MAX {
        bail!("blazec_encode_split_stream failed");
    }
    out.truncate(written);
    Ok(wrap_blxd(BlobEncoding::SplitStream, &out))
}

/// Decode a format-v2 blob into target tensor bytes.
pub fn decode_blob(
    blob: &[u8],
    base: Option<&[u8]>,
    expected_len: usize,
    _dtype: DType,
) -> Result<Vec<u8>> {
    let (enc, inner) = unwrap_blxd(blob)?;
    match enc {
        BlobEncoding::Raw => {
            if inner.len() != expected_len {
                bail!(
                    "raw blob length {} != expected {}",
                    inner.len(),
                    expected_len
                );
            }
            Ok(inner.to_vec())
        }
        BlobEncoding::ZstdOnly => {
            let out = decompress_zstd(inner, expected_len)?;
            if out.len() != expected_len {
                bail!("zstd-only decoded length mismatch");
            }
            Ok(out)
        }
        BlobEncoding::SparseByte => {
            let base = base.context("sparse_byte decode requires base tensor")?;
            decode_sparse_byte(base, inner)
        }
        BlobEncoding::SparseU16 => {
            let base = base.context("sparse_u16 decode requires base tensor")?;
            decode_sparse_u16(base, inner)
        }
        BlobEncoding::FullXor => {
            let base = base.context("full_xor decode requires base tensor")?;
            decode_full_xor(base, inner)
        }
        BlobEncoding::SplitStream => {
            let base = base.context("split_stream decode requires base tensor")?;
            decode_split_stream(base, inner)
        }
    }
}

fn decode_sparse_byte(base: &[u8], z: &[u8]) -> Result<Vec<u8>> {
    let inner = decompress_zstd(z, base.len().saturating_mul(2) + 1024)?;
    let mut cur = Cursor::new(&inner[..]);
    let tag = cur.read_u8()?;
    if tag != TAG_SPARSE_BYTE {
        bail!("sparse_byte: bad tag {tag:#x}");
    }
    let n_bytes = cur.read_u32::<LittleEndian>()? as usize;
    let _n_changed = cur.read_u32::<LittleEndian>()? as usize;
    if n_bytes != base.len() {
        bail!("sparse_byte: n_bytes mismatch");
    }
    let bitmap_len = (n_bytes + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_len];
    cur.read_exact(&mut bitmap)?;
    let values = &inner[cur.position() as usize..];
    let mut xor_buf = vec![0u8; n_bytes];
    let mut val_i = 0usize;
    for i in 0..n_bytes {
        if (bitmap[i / 8] >> (i % 8)) & 1 != 0 {
            let b = values
                .get(val_i)
                .context("sparse_byte: short xor stream")?;
            xor_buf[i] = *b;
            val_i += 1;
        }
    }
    if val_i != values.len() {
        bail!("sparse_byte: xor value count mismatch");
    }
    Ok(xor_apply(base, &xor_buf))
}

fn decode_sparse_u16(base: &[u8], z: &[u8]) -> Result<Vec<u8>> {
    let inner = decompress_zstd(z, base.len().saturating_mul(2) + 1024)?;
    let mut cur = Cursor::new(&inner[..]);
    let tag = cur.read_u8()?;
    if tag != TAG_SPARSE_U16 {
        bail!("sparse_u16: bad tag {tag:#x}");
    }
    let n_values = cur.read_u32::<LittleEndian>()? as usize;
    let _n_changed = cur.read_u32::<LittleEndian>()? as usize;
    if base.len() != n_values * 2 {
        bail!("sparse_u16: length mismatch");
    }
    let bitmap_len = (n_values + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_len];
    cur.read_exact(&mut bitmap)?;
    let values = &inner[cur.position() as usize..];
    let mut xor_buf = vec![0u8; base.len()];
    let mut vi = 0usize;
    for i in 0..n_values {
        if (bitmap[i / 8] >> (i % 8)) & 1 != 0 {
            if vi + 2 > values.len() {
                bail!("sparse_u16: short u16 xor stream");
            }
            let x = u16::from_le_bytes([values[vi], values[vi + 1]]);
            vi += 2;
            let o = i * 2;
            xor_buf[o..o + 2].copy_from_slice(&x.to_le_bytes());
        }
    }
    if vi != values.len() {
        bail!("sparse_u16: xor value count mismatch");
    }
    Ok(xor_apply(base, &xor_buf))
}

fn decode_full_xor(base: &[u8], z: &[u8]) -> Result<Vec<u8>> {
    let xor = decompress_zstd(z, base.len())?;
    if xor.len() != base.len() {
        bail!("full_xor: xor length mismatch");
    }
    Ok(xor_apply(base, &xor))
}

fn decode_split_stream(base: &[u8], payload: &[u8]) -> Result<Vec<u8>> {
    let capacity = base.len() + 64;
    let mut out = vec![0u8; capacity];
    let written = unsafe {
        codec_ffi::blazec_decode_split_stream(
            payload.as_ptr(),
            payload.len(),
            base.as_ptr(),
            base.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
    if written == usize::MAX {
        bail!("blazec_decode_split_stream failed");
    }
    out.truncate(written);
    Ok(out)
}

fn xor_apply(base: &[u8], xor: &[u8]) -> Vec<u8> {
    base.iter()
        .zip(xor.iter())
        .map(|(a, x)| a ^ x)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut out = vec![0u8; n];
        let mut s = seed;
        for b in &mut out {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (s >> 33) as u8;
        }
        out
    }

    #[test]
    fn roundtrip_raw_zstd_added() {
        let t = b"hello world payload".to_vec();
        let enc = encode_added(&t).unwrap();
        let dec = decode_blob(&enc, None, t.len(), DType::U8).unwrap();
        assert_eq!(dec, t);
    }

    #[test]
    fn roundtrip_full_xor_f32() {
        let base = rnd_bytes(64, 1);
        let mut target = base.clone();
        target[10..14].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let enc = encode_modified(&base, &target, DType::F32).unwrap();
        let dec = decode_blob(&enc, Some(&base), target.len(), DType::F32).unwrap();
        assert_eq!(dec, target);
    }

    #[test]
    fn roundtrip_sparse_byte_i8() {
        let base = vec![0u8; 100];
        let mut target = base.clone();
        target[7] = 3;
        target[99] = 9;
        let enc = encode_modified(&base, &target, DType::I8).unwrap();
        let dec = decode_blob(&enc, Some(&base), target.len(), DType::I8).unwrap();
        assert_eq!(dec, target);
    }

    #[test]
    fn roundtrip_split_stream_bf16() {
        let n = 128usize;
        let base = rnd_bytes(n * 2, 42);
        let mut target = base.clone();
        // flip a few BF16 slots
        for i in 0..20 {
            let o = (i * 11) % n * 2;
            target[o] ^= 0x0F;
        }
        let enc = encode_modified(&base, &target, DType::BF16).unwrap();
        let dec = decode_blob(&enc, Some(&base), target.len(), DType::BF16).unwrap();
        assert_eq!(dec, target);
    }
}
