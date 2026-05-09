# BlazEC (`libblazec`)

SplitStream encoding **is not implemented in Rust in this repository**. You must supply a prebuilt shared library per platform:

| Platform | File |
|----------|------|
| macOS Apple Silicon | `libblazec-aarch64-apple-darwin.dylib` |
| macOS x86_64 | `libblazec-x86_64-apple-darwin.dylib` |
| Linux x86_64 | `libblazec-x86_64-unknown-linux-gnu.so` |
| Windows x86_64 MSVC | `libblazec-x86_64-pc-windows-msvc.dll` |

`build.rs` copies the file for your compile target into `OUT_DIR` and links `-lblazec`. If the expected file is missing, **`cargo build` panics** with instructions — clone this repo alone is not enough to build until you add the binary.
