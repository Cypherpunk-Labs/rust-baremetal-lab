use burn_core as burn;

use alloc::vec::Vec;
use burn_core::{
    config::Config,
    module::Module,
    tensor::{activation::silu, backend::Backend, Int, Tensor},
};
use burn_nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};

#[derive(Config, Debug)]
pub struct SmolLmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[config(default = 1e-5)]
    pub rms_norm_eps: f64,
    #[config(default = 10000.0)]
    pub rope_theta: f64,
}

impl Default for SmolLmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 49152,
            hidden_size: 576,
            intermediate_size: 1536,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }
}

#[derive(Module, Debug)]
pub struct SmolLmModel<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub norm: RmsNorm<B>,
    #[module(skip)]
    pub config: SmolLmConfig,
}

impl<B: Backend> SmolLmModel<B> {
    pub fn new(config: &SmolLmConfig, device: &B::Device) -> Self {
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|_| TransformerBlock::new(config, device))
            .collect();
        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);
        Self {
            embed_tokens,
            layers,
            norm,
            config: config.clone(),
        }
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut hidden = self.embed_tokens.forward(input_ids);
        for layer in self.layers.iter() {
            hidden = layer.forward(hidden);
        }
        self.norm.forward(hidden)
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub input_layernorm: RmsNorm<B>,
    pub self_attn: Attention<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub mlp: Mlp<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(config: &SmolLmConfig, device: &B::Device) -> Self {
        let input_layernorm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);
        let self_attn = Attention::new(config, device);
        let post_attention_layernorm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);
        let mlp = Mlp::new(config, device);
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(x) + residual;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x) + residual;
        x
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &SmolLmConfig, device: &B::Device) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let qkv_dim = head_dim * config.num_key_value_heads;
        let q_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(false)
            .init(device);
        let k_proj = LinearConfig::new(config.hidden_size, qkv_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(config.hidden_size, qkv_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(qkv_dim, config.hidden_size)
            .with_bias(false)
            .init(device);
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Placeholder: structural projections only.
        // RoPE and scaled dot-product attention are TODO.
        let q = self.q_proj.forward(x);
        self.o_proj.forward(q)
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &SmolLmConfig, device: &B::Device) -> Self {
        let gate_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(false)
            .init(device);
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down_proj.forward(silu(self.gate_proj.forward(x.clone())) * self.up_proj.forward(x))
    }
}