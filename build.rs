fn main() {
    let target = std::env::var("TARGET").unwrap();

    let lib_name = if target.contains("apple") {
        if target.contains("aarch64") {
            "libblazec-aarch64-apple-darwin.dylib"
        } else {
            "libblazec-x86_64-apple-darwin.dylib"
        }
    } else if target.contains("windows") {
        "libblazec-x86_64-pc-windows-msvc.dll"
    } else {
        "libblazec-x86_64-unknown-linux-gnu.so"
    };

    let codec_dir = std::path::Path::new("codec");
    let lib_path = codec_dir.join(lib_name);

    if !lib_path.exists() {
        panic!(
            "BlazEC codec binary not found at {}.\n\
             Download the correct `libblazec` for your platform ({}) from the project releases page and place it under codec/",
            lib_path.display(),
            target
        );
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = std::path::Path::new(&out_dir);
    let dst = if target.contains("windows") {
        out_dir.join("blazec.dll")
    } else if target.contains("apple") {
        out_dir.join("libblazec.dylib")
    } else {
        out_dir.join("libblazec.so")
    };

    std::fs::copy(&lib_path, &dst).unwrap_or_else(|e| {
        panic!(
            "failed to copy {} to {}: {e}",
            lib_path.display(),
            dst.display()
        );
    });

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=blazec");
    println!("cargo:rerun-if-changed=codec/");
}
