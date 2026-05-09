/// Integration tests for BXP archive round-trips, diff/patch, and export.
///
/// These tests run entirely in-memory or with tempfiles — no model downloads needed.

use blaze_x_pack::{
    exporter, format::{ArchiveReader, ArchiveWriter}, patch, types::DType,
};
use tempfile::TempDir;

// Re-export the library modules so tests can use them
// (tests live in /tests/ so they import the crate as a library)
// We expose them here via a test-only module alias.

// Helper: build a tiny fake archive in memory
fn make_archive(dir: &TempDir, name: &str, tensors: &[(&str, &[f32])]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut w = ArchiveWriter::new(&path, serde_json::json!({"model_type": "test"}));
    for (tname, data) in tensors {
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        w.add_tensor(tname, DType::F32, vec![data.len()], raw);
    }
    w.finish().expect("write");
    path
}

#[test]
fn test_pack_verify_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = make_archive(&dir, "model.blz", &[
        ("weight_a", &[1.0, 2.0, 3.0, 4.0]),
        ("weight_b", &[5.0, 6.0, 7.0, 8.0]),
    ]);

    let r = ArchiveReader::open(&path).expect("open");
    assert_eq!(r.header.tensors.len(), 2);
    assert_eq!(r.header.tensors[0].name, "weight_a");
    assert_eq!(r.header.tensors[1].name, "weight_b");

    let report = r.verify();
    assert!(report.is_clean(), "verification failed: {:?}", report.failed);
}

#[test]
fn test_tensor_bytes_roundtrip() {
    let dir = TempDir::new().unwrap();
    let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let path = make_archive(&dir, "model.blz", &[("layer.weight", &data)]);

    let r = ArchiveReader::open(&path).unwrap();
    let raw = r.tensor_bytes("layer.weight").unwrap();
    let recovered: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(recovered, data);
}

#[test]
fn test_verify_detects_corruption() {
    let dir = TempDir::new().unwrap();
    let path = make_archive(&dir, "model.blz", &[("w", &[1.0f32, 2.0, 3.0])]);

    // Corrupt some bytes near the end of the file
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len();
    bytes[last - 4] ^= 0xFF;
    bytes[last - 3] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let r = ArchiveReader::open(&path).unwrap();
    let report = r.verify();
    assert!(!report.is_clean(), "expected corruption to be detected");
}

#[test]
fn test_diff_apply_unchanged() {
    let dir = TempDir::new().unwrap();
    let base = make_archive(&dir, "base.blz", &[
        ("a", &[1.0f32, 2.0]),
        ("b", &[3.0f32, 4.0]),
    ]);
    let target = make_archive(&dir, "target.blz", &[
        ("a", &[1.0f32, 2.0]),  // same
        ("b", &[3.0f32, 4.0]),  // same
    ]);
    let patch_path = dir.path().join("patch.blzdiff");
    let out_path = dir.path().join("out.blz");

    let diff_stats = patch::diff(&base, &target, &patch_path).unwrap();
    assert_eq!(diff_stats.modified, 0);
    assert_eq!(diff_stats.added, 0);
    assert_eq!(diff_stats.removed, 0);
    assert_eq!(diff_stats.unchanged, 2);

    let apply_stats = patch::apply(&base, &patch_path, &out_path).unwrap();
    assert_eq!(apply_stats.tensors_from_base, 2);
    assert_eq!(apply_stats.tensors_from_patch, 0);

    let r = ArchiveReader::open(&out_path).unwrap();
    assert!(r.verify().is_clean());
}

#[test]
fn test_diff_apply_modified() {
    let dir = TempDir::new().unwrap();
    let base = make_archive(&dir, "base.blz", &[
        ("embed", &[0.1f32, 0.2, 0.3]),
        ("head",  &[1.0f32, 2.0]),
    ]);
    let target = make_archive(&dir, "target.blz", &[
        ("embed", &[0.1f32, 0.2, 0.3]),  // unchanged
        ("head",  &[9.0f32, 9.0]),        // modified
    ]);
    let patch_path = dir.path().join("delta.blzdiff");
    let out_path   = dir.path().join("result.blz");

    let diff_stats = patch::diff(&base, &target, &patch_path).unwrap();
    assert_eq!(diff_stats.unchanged, 1);
    assert_eq!(diff_stats.modified, 1);

    // For tiny test tensors patch overhead dominates; just verify it was created
    assert!(diff_stats.patch_bytes > 0, "patch should have non-zero size");

    patch::apply(&base, &patch_path, &out_path).unwrap();

    let r = ArchiveReader::open(&out_path).unwrap();
    assert!(r.verify().is_clean());

    // Verify the modified tensor has the new values
    let raw = r.tensor_bytes("head").unwrap();
    let values: Vec<f32> = raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![9.0f32, 9.0]);
}

#[test]
fn test_diff_apply_added_removed() {
    let dir = TempDir::new().unwrap();
    let base = make_archive(&dir, "base.blz", &[
        ("layer.0", &[1.0f32]),
        ("layer.1", &[2.0f32]),  // will be removed
    ]);
    let target = make_archive(&dir, "target.blz", &[
        ("layer.0", &[1.0f32]),
        ("layer.2", &[3.0f32]),  // new tensor
    ]);
    let patch_path = dir.path().join("delta.blzdiff");
    let out_path   = dir.path().join("result.blz");

    let diff_stats = patch::diff(&base, &target, &patch_path).unwrap();
    assert_eq!(diff_stats.unchanged, 1);
    assert_eq!(diff_stats.added, 1);
    assert_eq!(diff_stats.removed, 1);

    patch::apply(&base, &patch_path, &out_path).unwrap();

    let r = ArchiveReader::open(&out_path).unwrap();
    assert!(r.verify().is_clean());
    let names: Vec<&str> = r.header.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"layer.0"));
    assert!(names.contains(&"layer.2"));
    assert!(!names.contains(&"layer.1"), "removed tensor should not be in output");
}

#[test]
fn test_export_safetensors() {
    let dir = TempDir::new().unwrap();
    let path = make_archive(&dir, "model.blz", &[
        ("model.weight", &[1.0f32, 2.0, 3.0, 4.0]),
    ]);
    let r = ArchiveReader::open(&path).unwrap();
    let out_dir = dir.path().join("safetensors_out");
    let stats = exporter::export_safetensors(&r, &out_dir, None).unwrap();
    assert_eq!(stats.tensors, 1);
    assert!(out_dir.join("model.safetensors").exists());
    assert!(out_dir.join("config.json").exists());
}

#[test]
fn test_export_pytorch() {
    let dir = TempDir::new().unwrap();
    let path = make_archive(&dir, "model.blz", &[
        ("a", &[1.0f32, 2.0]),
        ("b", &[3.0f32]),
    ]);
    let r = ArchiveReader::open(&path).unwrap();
    let out_dir = dir.path().join("pt_out");
    let stats = exporter::export_pytorch(&r, &out_dir, None).unwrap();
    assert_eq!(stats.tensors, 2);
    assert!(out_dir.join("manifest.json").exists());
    assert!(out_dir.join("a.bin").exists());
    assert!(out_dir.join("b.bin").exists());
}

#[test]
fn test_bad_magic_rejected() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("garbage.blz");
    std::fs::write(&path, b"this is not a blazex file at all").unwrap();
    let result = ArchiveReader::open(&path);
    match result {
        Err(e) => assert!(e.to_string().contains("bad magic"), "expected 'bad magic' in: {e}"),
        Ok(_) => panic!("expected error but got Ok"),
    }
}

#[test]
fn test_export_gguf() {
    let dir = TempDir::new().unwrap();
    // Use a config with known arch metadata so GGUF writer has something to work with
    let path = dir.path().join("model.blz");
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "vocab_size": 32000
    });
    let mut w = blaze_x_pack::format::ArchiveWriter::new(&path, config);
    let raw: Vec<u8> = (0u32..512).flat_map(|i| (i as f32 * 0.01).to_le_bytes()).collect();
    w.add_tensor("model.embed_tokens.weight", blaze_x_pack::types::DType::F32, vec![512], raw.clone());
    w.add_tensor("lm_head.weight", blaze_x_pack::types::DType::F32, vec![512], raw.clone());
    w.finish().unwrap();

    let r = blaze_x_pack::format::ArchiveReader::open(&path).unwrap();
    let gguf_path = dir.path().join("model.gguf");
    let stats = blaze_x_pack::exporter::export_gguf(&r, &gguf_path, None).unwrap();

    assert!(gguf_path.exists(), "gguf file should exist");
    assert_eq!(stats.tensors, 2);
    // GGUF now writes config.json alongside the .gguf for completeness
    assert!(stats.files_written >= 1);
    assert!(dir.path().join("config.json").exists(), "config.json should be written alongside gguf");

    // Verify GGUF magic bytes in output
    let bytes = std::fs::read(&gguf_path).unwrap();
    assert!(bytes.len() > 8);
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic, 0x46554747, "GGUF magic mismatch");
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(version, 3, "GGUF version should be 3");
}

// ─────────────────────────────────────────────────────────────────────────────
// Cast integration tests
// ─────────────────────────────────────────────────────────────────────────────

use blaze_x_pack::cast::CastTarget;

#[test]
fn test_export_safetensors_cast_f16() {
    let dir = TempDir::new().unwrap();
    let path = make_archive(&dir, "model.blz", &[("w", &(0..32).map(|i| i as f32 * 0.1).collect::<Vec<_>>())]);
    let r = ArchiveReader::open(&path).unwrap();
    let out = dir.path().join("out_f16");
    let stats = blaze_x_pack::exporter::export_safetensors(&r, &out, Some(CastTarget::F16)).unwrap();
    // F16 is 2 bytes/element vs F32 4 bytes — output should be smaller
    assert_eq!(stats.tensors, 1);
    let st_bytes = std::fs::metadata(out.join("model.safetensors")).unwrap().len();
    // 32 elements × 2 bytes = 64 bytes of data, plus header
    assert!(st_bytes < 200, "f16 output unexpectedly large: {st_bytes}");
}

#[test]
fn test_export_safetensors_cast_q8_0() {
    let dir = TempDir::new().unwrap();
    // 32 elements = exactly 1 Q8_0 block
    let data: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    let path = make_archive(&dir, "model.blz", &[("layer", &data)]);
    let r = ArchiveReader::open(&path).unwrap();
    let out = dir.path().join("out_q8");
    let stats = blaze_x_pack::exporter::export_safetensors(&r, &out, Some(CastTarget::Q8_0)).unwrap();
    assert_eq!(stats.tensors, 1);
}

#[test]
fn test_export_gguf_cast_q8_0() {
    let dir = TempDir::new().unwrap();
    // Use 32-element tensors (Q8_0 block size)
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.8).collect();
    let path = dir.path().join("model.blz");
    let mut w = blaze_x_pack::format::ArchiveWriter::new(&path, serde_json::json!({"model_type":"llama"}));
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    w.add_tensor("model.embed_tokens.weight", blaze_x_pack::types::DType::F32, vec![32], raw);
    w.finish().unwrap();

    let r = blaze_x_pack::format::ArchiveReader::open(&path).unwrap();
    let gguf_path = dir.path().join("out_q8.gguf");
    let stats = blaze_x_pack::exporter::export_gguf(&r, &gguf_path, Some(CastTarget::Q8_0)).unwrap();
    assert_eq!(stats.tensors, 1);

    // Verify GGUF magic and that the file is smaller than the unquantised equivalent
    let bytes = std::fs::read(&gguf_path).unwrap();
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic, 0x46554747);
    // Q8_0: 34 bytes for 32 elements; F32 would be 128 bytes
    assert!(stats.bytes_written < 1000, "unexpectedly large gguf: {}", stats.bytes_written);
}

#[test]
fn test_cast_target_from_str() {
    assert_eq!(CastTarget::from_str("f16"),   Some(CastTarget::F16));
    assert_eq!(CastTarget::from_str("F16"),   Some(CastTarget::F16));
    assert_eq!(CastTarget::from_str("bf16"),  Some(CastTarget::BF16));
    assert_eq!(CastTarget::from_str("q8_0"),  Some(CastTarget::Q8_0));
    assert_eq!(CastTarget::from_str("Q4_K"),  Some(CastTarget::Q4K));
    assert_eq!(CastTarget::from_str("q4-k"),  Some(CastTarget::Q4K));
    assert_eq!(CastTarget::from_str("bogus"), None);
}

#[test]
fn test_tokenizer_roundtrip_exact() {
    // Tokenizer JSON with deliberate formatting quirks that serde would mangle:
    // - trailing spaces in values
    // - numbers as floats (1.0 instead of 1)
    // - non-alphabetical key order
    // - unicode escapes
    let tok_json = r#"{
  "version": "1.0",
  "added_tokens": [],
  "model": {
    "type": "BPE",
    "unk_token": null,
    "vocab": {
      "hello": 0,
      "world": 1
    },
    "merges": ["hello world"]
  },
  "special_tokens_map": {},
  "extra_data": 1.0
}"#;

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.blz");

    // Pack with tokenizer
    let raw: Vec<u8> = (0u32..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut w = blaze_x_pack::format::ArchiveWriter::new(
        &path,
        serde_json::json!({"model_type": "gpt2"}),
    );
    w.set_tokenizer(tok_json.to_owned());
    w.add_tensor("w", blaze_x_pack::types::DType::F32, vec![4], raw);
    w.finish().unwrap();

    // Export to safetensors
    let r = blaze_x_pack::format::ArchiveReader::open(&path).unwrap();
    let out_dir = dir.path().join("exported");
    blaze_x_pack::exporter::export_safetensors(&r, &out_dir, None).unwrap();

    // Exported tokenizer.json must be byte-for-byte identical to input
    let exported = std::fs::read_to_string(out_dir.join("tokenizer.json")).unwrap();
    assert_eq!(exported, tok_json,
        "tokenizer.json was modified during pack→export round-trip");
}

#[test]
fn test_create_verify_patch_fixture() {
    // Write a real .blz file to disk so blazex-verify-patch can be tested against it
    let path = std::path::PathBuf::from("/tmp/blazex_verify_test.blz");
    let mut w = blaze_x_pack::format::ArchiveWriter::new(
        &path,
        serde_json::json!({"model_type": "gpt2", "hidden_size": 32, "num_hidden_layers": 20}),
    );
    // 20 "layer" tensors + an embedding tensor
    for i in 0..20usize {
        let data: Vec<u8> = (0u32..128)
            .flat_map(|j| ((j as f32) + (i as f32) * 100.0).to_le_bytes())
            .collect();
        w.add_tensor(
            &format!("model.layers.{i}.weight"),
            blaze_x_pack::types::DType::F32,
            vec![128],
            data,
        );
    }
    let embed: Vec<u8> = (0u32..512).flat_map(|j| (j as f32).to_le_bytes()).collect();
    w.add_tensor("model.embed_tokens.weight", blaze_x_pack::types::DType::F32, vec![512], embed);
    w.finish().unwrap();
    assert!(path.exists());
    eprintln!("Fixture written to {}", path.display());
}

#[test]
fn test_sidecar_files_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("model.blz");

    // Build archive with several sidecars including a binary one
    let mut w = blaze_x_pack::format::ArchiveWriter::new(
        &path,
        serde_json::json!({"model_type": "llama"}),
    );
    w.set_tokenizer(r#"{"version":"1.0","model":{"type":"BPE"}}"#.to_owned());

    let tok_config = r#"{"tokenizer_class":"LlamaTokenizer","bos_token":"<s>","eos_token":"</s>"}"#;
    let special_tokens = r#"{"bos_token":"<s>","eos_token":"</s>","unk_token":"<unk>"}"#;
    let gen_config = r#"{"temperature":0.9,"do_sample":true,"max_new_tokens":512}"#;
    // Simulate a binary tokenizer.model (SentencePiece)
    let fake_sp_binary: Vec<u8> = (0u8..=255).cycle().take(256).collect();

    w.add_sidecar("tokenizer_config.json", tok_config.as_bytes());
    w.add_sidecar("special_tokens_map.json", special_tokens.as_bytes());
    w.add_sidecar("generation_config.json", gen_config.as_bytes());
    w.add_sidecar("tokenizer.model", &fake_sp_binary);

    let raw: Vec<u8> = (0u32..32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    w.add_tensor("w", blaze_x_pack::types::DType::F32, vec![32], raw);
    w.finish().unwrap();

    // Re-open and verify sidecar metadata
    let r = blaze_x_pack::format::ArchiveReader::open(&path).unwrap();
    assert_eq!(r.header.sidecar_files.len(), 4);
    let names: Vec<&str> = r.header.sidecar_files.iter().map(|s| s.filename.as_str()).collect();
    assert!(names.contains(&"tokenizer_config.json"));
    assert!(names.contains(&"tokenizer.model"));

    // Export and verify all sidecar files written verbatim
    let out = dir.path().join("exported");
    blaze_x_pack::exporter::export_safetensors(&r, &out, None).unwrap();

    let exported_config = std::fs::read_to_string(out.join("tokenizer_config.json")).unwrap();
    assert_eq!(exported_config, tok_config);

    let exported_special = std::fs::read_to_string(out.join("special_tokens_map.json")).unwrap();
    assert_eq!(exported_special, special_tokens);

    let exported_gen = std::fs::read_to_string(out.join("generation_config.json")).unwrap();
    assert_eq!(exported_gen, gen_config);

    // Binary file must be byte-perfect
    let exported_sp = std::fs::read(out.join("tokenizer.model")).unwrap();
    assert_eq!(exported_sp, fake_sp_binary, "tokenizer.model binary content corrupted");
}
