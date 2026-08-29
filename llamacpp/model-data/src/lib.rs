//! Embedded model weights for the on-device chat demo.
#![no_std]
//!
//! `MODEL_BYTES` is the raw SmolLM-135M F32 `model.safetensors` blob, embedded
//! at compile time. The `model` crate parses this buffer (safetensors format)
//! and feeds the tensors into ggml. Single source of truth:
//! `../../../burn/model-builder/model.safetensors`.

pub const MODEL_BYTES: &[u8] = include_bytes!("../../../burn/model-builder/model.safetensors");

/// Compact GPT-2 byte-level BPE tokenizer blob for SmolLM, produced by
/// `burn/model-builder`'s `tokenizer-builder`. Parsed by `kernel::tokenizer`.
pub const TOKENIZER_BYTES: &[u8] = include_bytes!("tokenizer.bin");
