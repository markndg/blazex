//! FFI to the proprietary BlazEC SplitStream codec (`libblazec`).
//! No SplitStream implementation exists in this repository — only these declarations.
//! The shared library must be placed under `codec/` (see `build.rs`).

#[link(name = "blazec", kind = "dylib")]
unsafe extern "C" {
    /// Encode XOR bytes using SplitStream. Returns bytes written, or `usize::MAX` on error.
    pub fn blazec_encode_split_stream(
        xor_ptr: *const u8,
        xor_len: usize,
        out_ptr: *mut u8,
        out_capacity: usize,
    ) -> usize;

    /// Decode SplitStream payload with base tensor bytes into target. Returns bytes written,
    /// or `usize::MAX` on error.
    pub fn blazec_decode_split_stream(
        payload_ptr: *const u8,
        payload_len: usize,
        base_ptr: *const u8,
        base_len: usize,
        out_ptr: *mut u8,
        out_capacity: usize,
    ) -> usize;
}
