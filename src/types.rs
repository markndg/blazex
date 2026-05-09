//! Shared types used across all modules

use serde::{Deserialize, Serialize};

/// Data type of a tensor — mirrors SafeTensors / GGUF conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I32,
    I64,
    U8,
    U16,
    U32,
    Bool,
}

impl DType {
    /// Bytes per element
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
            DType::I64 => 8,
        }
    }

    /// Canonical string name (matches SafeTensors)
    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I8 => "I8",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::U16 => "U16",
            DType::U32 => "U32",
            DType::Bool => "BOOL",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "F32" | "FLOAT32" => Some(DType::F32),
            "F16" | "FLOAT16" | "HALF" => Some(DType::F16),
            "BF16" | "BFLOAT16" => Some(DType::BF16),
            "I8" | "INT8" => Some(DType::I8),
            "I32" | "INT32" => Some(DType::I32),
            "I64" | "INT64" => Some(DType::I64),
            "U8" | "UINT8" => Some(DType::U8),
            "U16" | "UINT16" => Some(DType::U16),
            "U32" | "UINT32" => Some(DType::U32),
            "BOOL" => Some(DType::Bool),
            _ => None,
        }
    }
}

/// Lightweight metadata about a tensor — used in loader / exporter
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

#[allow(dead_code)]
impl TensorMeta {
    pub fn byte_size(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.byte_size()
    }
}
