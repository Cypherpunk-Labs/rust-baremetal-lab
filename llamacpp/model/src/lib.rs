#![no_std]
extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use alloc::format;

pub mod safetensors;
pub mod forward;

/// SmolLM-135M architecture constants (identical to the Burn kernel's
/// `SmolLmConfig`). Used both on-device and by the host parity harness.
#[derive(Clone, Copy)]
pub struct SmolLmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl SmolLmConfig {
    pub const DEFAULT: SmolLmConfig = SmolLmConfig {
        vocab_size: 49152,
        hidden_size: 576,
        intermediate_size: 1536,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    };

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}

/// Resolved weights for one transformer block, owned F32 vectors (copied once
/// from the embedded safetensors buffer at parse time).
pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

pub struct SmolLmWeights {
    pub config: SmolLmConfig,
    pub embed_tokens: Vec<f32>,
    pub norm: Vec<f32>,
    pub layers: Vec<LayerWeights>,
}

impl SmolLmWeights {
    /// Build a typed view over the parsed safetensors blob. Tensor names follow
    /// the HuggingFace SmolLM layout; lm_head is tied to embed_tokens.
    pub fn from_safetensors(st: &safetensors::SafeTensors, cfg: SmolLmConfig) -> Self {
        let get = |name: &str| st.tensor_f32(name);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights {
                input_layernorm: get(&format!("model.layers.{}.input_layernorm.weight", i)),
                post_attention_layernorm: get(&format!(
                    "model.layers.{}.post_attention_layernorm.weight",
                    i
                )),
                q_proj: get(&format!("model.layers.{}.self_attn.q_proj.weight", i)),
                k_proj: get(&format!("model.layers.{}.self_attn.k_proj.weight", i)),
                v_proj: get(&format!("model.layers.{}.self_attn.v_proj.weight", i)),
                o_proj: get(&format!("model.layers.{}.self_attn.o_proj.weight", i)),
                gate_proj: get(&format!("model.layers.{}.mlp.gate_proj.weight", i)),
                up_proj: get(&format!("model.layers.{}.mlp.up_proj.weight", i)),
                down_proj: get(&format!("model.layers.{}.mlp.down_proj.weight", i)),
            });
        }
        SmolLmWeights {
            config: cfg,
            embed_tokens: get("model.embed_tokens.weight"),
            norm: get("model.norm.weight"),
            layers,
        }
    }
}

/// Format a float without libm (matches the kernel's diag formatter).
pub fn format_f32(v: f32) -> String {
    if v < 0.0 {
        return alloc::format!("-{}", format_f32(-v));
    }
    let i = v as u64;
    let frac = ((v - i as f32) * 1000.0) as u64;
    alloc::format!("{}.{:03}", i, frac)
}
