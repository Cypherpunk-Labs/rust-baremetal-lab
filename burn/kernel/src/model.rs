use burn_core as burn;

use alloc::vec;
use alloc::vec::Vec;
use burn_core::{
    config::Config,
    module::Module,
    tensor::{activation::silu, backend::Backend, Bool, Int, Tensor},
};
use burn_nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};

fn isqrt(n: usize) -> usize {
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

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
        let hidden = self.norm.forward(hidden);

        // Tied LM head: logits = hidden @ embed_tokens.weight^T (tie_word_embeddings=true)
        let weight = self.embed_tokens.weight.val();
        let [b, _s, _] = hidden.shape().dims();
        let weight = weight
            .transpose()
            .unsqueeze_dim::<3>(0)
            .expand([b, self.config.hidden_size, self.config.vocab_size]);
        hidden.matmul(weight)
    }

    pub fn new_cache(&self) -> KvCache<B> {
        KvCache::new(self.config.num_hidden_layers)
    }

    /// Autoregressive step with a KV cache. Processes only the new tokens in
    /// `input_ids` (positions `cache.len()..cache.len()+s`) and appends their
    /// keys/values to the cache. Returns logits `[b, s, vocab]` for the new
    /// block; generation picks the last position.
    pub fn forward_step(&self, input_ids: Tensor<B, 2, Int>, cache: &mut KvCache<B>) -> Tensor<B, 3> {
        let start_pos = cache.len;
        let mut hidden = self.embed_tokens.forward(input_ids);
        let [b, _s, _] = hidden.shape().dims();
        for (i, layer) in self.layers.iter().enumerate() {
            let (h, k_all, v_all) = layer.forward_step(
                hidden,
                cache.keys[i].as_ref(),
                cache.values[i].as_ref(),
                start_pos,
            );
            hidden = h;
            cache.keys[i] = Some(k_all);
            cache.values[i] = Some(v_all);
        }
        cache.len += hidden.shape().dims::<3>()[1];
        let hidden = self.norm.forward(hidden);

        let weight = self.embed_tokens.weight.val();
        let weight = weight
            .transpose()
            .unsqueeze_dim::<3>(0)
            .expand([b, self.config.hidden_size, self.config.vocab_size]);
        hidden.matmul(weight)
    }
}

/// Per-layer cached keys/values for autoregressive generation.
pub struct KvCache<B: Backend> {
    pub keys: Vec<Option<Tensor<B, 4>>>,
    pub values: Vec<Option<Tensor<B, 4>>>,
    pub len: usize,
}

impl<B: Backend> KvCache<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: (0..num_layers).map(|_| None).collect(),
            values: (0..num_layers).map(|_| None).collect(),
            len: 0,
        }
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

    /// Cached forward: attend over cached K/V plus the new block's K/V, then
    /// continue with the MLP for the new positions only. Returns the new hidden
    /// state along with the merged keys/values to store back into the cache.
    pub fn forward_step(
        &self,
        x: Tensor<B, 3>,
        cache_k: Option<&Tensor<B, 4>>,
        cache_v: Option<&Tensor<B, 4>>,
        start_pos: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let (x, k, v) = self.self_attn.forward_step(x, cache_k, cache_v, start_pos);
        let x = x + residual;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x) + residual;
        (x, k, v)
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub num_attention_heads: usize,
    #[module(skip)]
    pub num_key_value_heads: usize,
    #[module(skip)]
    pub rope_theta: f64,
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
            head_dim,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            rope_theta: config.rope_theta,
        }
    }

    fn rope_freqs(&self, start: usize, len: usize, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let half = self.head_dim / 2;

        // inv_freq[i] = 1 / theta^(2i/head_dim), i in 0..half
        let idx = Tensor::<B, 1, Int>::arange(0..half as i64, device).float();
        let expo = idx.mul_scalar(-2.0) / (self.head_dim as f32);
        let theta_t = Tensor::<B, 1>::full([half], self.rope_theta as f32, device);
        let inv_freq = theta_t.powf(expo);

        // positions (absolute) outer product inv_freq -> [len, half]
        let pos = Tensor::<B, 1, Int>::arange(start as i64..(start + len) as i64, device).float();
        let pos = pos.unsqueeze_dim::<2>(1); // [len, 1]
        let inv_freq = inv_freq.unsqueeze_dim::<2>(0); // [1, half]
        let freqs = pos.matmul(inv_freq); // [len, half]

        (freqs.clone().cos(), freqs.sin())
    }

    fn apply_rope(
        &self,
        x: Tensor<B, 4>,
        cos: &Tensor<B, 2>,
        sin: &Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let [b, s, heads, _] = x.shape().dims();
        let half = self.head_dim / 2;

        // Broadcast cos/sin [s, half] -> [b, s, heads, half]
        let cos = cos
            .clone()
            .unsqueeze_dim::<3>(0) // [1, s, half]
            .unsqueeze_dim::<4>(2) // [1, s, 1, half]
            .expand([b, s, heads, half]);
        let sin = sin
            .clone()
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2)
            .expand([b, s, heads, half]);

        let x1 = x.clone().narrow(3, 0, half);
        let x2 = x.narrow(3, half, half);

        // rotate_half: [x1*cos - x2*sin, x1*sin + x2*cos]
        let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let out2 = x1 * sin + x2 * cos;
        Tensor::cat(vec![out1, out2], 3)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.shape().dims();
        let device = x.device();
        let head_dim = self.head_dim;
        let heads = self.num_attention_heads;
        let kv_heads = self.num_key_value_heads;
        let group = heads / kv_heads;

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([b, s, heads, head_dim]);
        let k = k.reshape([b, s, kv_heads, head_dim]);
        let v = v.reshape([b, s, kv_heads, head_dim]);

        let (cos, sin) = self.rope_freqs(0, s, &device);
        let q = self.apply_rope(q, &cos, &sin);
        let k = self.apply_rope(k, &cos, &sin);

        // GQA: repeat k/v heads `group` times -> [b, s, heads, head_dim]
        let k = k
            .unsqueeze_dim::<5>(3)
            .expand([b, s, kv_heads, group, head_dim])
            .reshape([b, s, heads, head_dim]);
        let v = v
            .unsqueeze_dim::<5>(3)
            .expand([b, s, kv_heads, group, head_dim])
            .reshape([b, s, heads, head_dim]);

        // Move heads before sequence so matmul batches over [b, heads]:
        // q [b, heads, s, hd], k_t [b, heads, hd, s]
        let q = q.swap_dims(1, 2);
        let k_t = k.swap_dims(1, 2).swap_dims(2, 3);
        let v = v.swap_dims(1, 2);

        // scores = q @ k^T / sqrt(head_dim)  -> [b, heads, s, s]
        let scores = q.matmul(k_t).div_scalar(isqrt(head_dim) as f32);

        // causal mask: key_pos > query_pos -> -inf
        let mask = Tensor::<B, 2, Bool>::tril_mask([s, s], 0, &device);
        let mask = mask.reshape([1, 1, s, s]).expand([b, heads, s, s]);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let attn = burn_core::tensor::activation::softmax(scores, 3);
        let out = attn.matmul(v); // [b, heads, s, hd]
        let out = out.swap_dims(1, 2); // [b, s, heads, hd]
        let out = out.reshape([b, s, heads * head_dim]);
        self.o_proj.forward(out)
    }

    /// Cached forward. `cache_k`/`cache_v` hold RoPE'd keys/values for absolute
    /// positions `0..start_pos`. Computes q/k/v for the new block at
    /// `start_pos..start_pos+s`, appends k/v to the cache, and attends the new
    /// queries over all cached keys. Returns the output plus the merged
    /// keys/values to store back into the cache.
    pub fn forward_step(
        &self,
        x: Tensor<B, 3>,
        cache_k: Option<&Tensor<B, 4>>,
        cache_v: Option<&Tensor<B, 4>>,
        start_pos: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let [b, s, _] = x.shape().dims();
        let device = x.device();
        let head_dim = self.head_dim;
        let heads = self.num_attention_heads;
        let kv_heads = self.num_key_value_heads;
        let group = heads / kv_heads;

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        let q = q.reshape([b, s, heads, head_dim]);
        let k = k.reshape([b, s, kv_heads, head_dim]);
        let v = v.reshape([b, s, kv_heads, head_dim]);

        let (cos, sin) = self.rope_freqs(start_pos, s, &device);
        let q = self.apply_rope(q, &cos, &sin);
        let k = self.apply_rope(k, &cos, &sin);

        let cache_len = cache_k.map(|t| t.shape().dims::<4>()[1]).unwrap_or(0);
        // Caches hold K/V in kv_heads form [b, len, kv_heads, head_dim].
        let k_all = match cache_k {
            Some(c) => Tensor::cat(vec![c.clone(), k.clone()], 1),
            None => k.clone(),
        };
        let v_all = match cache_v {
            Some(c) => Tensor::cat(vec![c.clone(), v.clone()], 1),
            None => v.clone(),
        };

        // GQA: repeat k/v heads `group` times for attention -> [b, cache_len+s, heads, head_dim]
        let k_all_exp = k_all
            .clone()
            .unsqueeze_dim::<5>(3)
            .expand([b, cache_len + s, kv_heads, group, head_dim])
            .reshape([b, cache_len + s, heads, head_dim]);
        let v_all_exp = v_all
            .clone()
            .unsqueeze_dim::<5>(3)
            .expand([b, cache_len + s, kv_heads, group, head_dim])
            .reshape([b, cache_len + s, heads, head_dim]);

        // q [b, heads, s, hd], k_t [b, heads, hd, cache_len+s]
        let q = q.swap_dims(1, 2);
        let k_t = k_all_exp.swap_dims(1, 2).swap_dims(2, 3);
        let v = v_all_exp.swap_dims(1, 2);

        let scores = q.matmul(k_t).div_scalar(isqrt(head_dim) as f32);

        // causal mask: key_pos > query_pos -> -inf; the diagonal offset equals
        // the cache length so past positions are always attended.
        let mask = Tensor::<B, 2, Bool>::tril_mask([s, cache_len + s], cache_len as i64, &device);
        let mask = mask
            .reshape([1, 1, s, cache_len + s])
            .expand([b, heads, s, cache_len + s]);
        let scores = scores.mask_fill(mask, f32::NEG_INFINITY);

        let attn = burn_core::tensor::activation::softmax(scores, 3);
        let out = attn.matmul(v); // [b, heads, s, hd]
        let out = out.swap_dims(1, 2); // [b, s, heads, hd]
        let out = out.reshape([b, s, heads * head_dim]);
        (self.o_proj.forward(out), k_all, v_all)
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