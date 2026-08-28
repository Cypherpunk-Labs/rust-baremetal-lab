//! Hand-written SmolLM-135M forward pass built on the ggml CPU backend.
//!
//! ggml semantics used:
//!   * ggml_mul_mat(a, b) computes a^T @ b. Result shape is
//!     { a.ne[1], b.ne[1], b.ne[2], b.ne[3] } and requires a.ne[0] == b.ne[0].
//!   * Therefore every linear weight is stored as W^T: ggml tensor
//!     [ne0 = in_dim, ne1 = out_dim] whose data is the transpose of the HF
//!     weight [out_dim, in_dim]. Then ggml_mul_mat(Wt, x) == W @ x.
//!   * For the attention matmuls we arrange the per-head tensors as
//!     [head_dim, seq, n_head] so that ne[0] == head_dim is the summed dim.

use core::ffi::c_int;
use alloc::vec::Vec;
use ggml_sys::*;
use libm::sqrtf;

use crate::{SmolLmConfig, SmolLmWeights};

pub struct Model {
    ctx: *mut ggml_context,
    cfg: SmolLmConfig,
    embed: *mut ggml_tensor, // [hidden, vocab] (W^T of HF [vocab, hidden])
    norm: *mut ggml_tensor,  // [hidden]
    layers: Vec<LayerTensors>,
}

struct LayerTensors {
    input_ln: *mut ggml_tensor, // [hidden]
    post_ln: *mut ggml_tensor,  // [hidden]
    q: *mut ggml_tensor,        // [hidden, hidden]  (W^T)
    k: *mut ggml_tensor,        // [hidden, kv_dim]  (W^T)
    v: *mut ggml_tensor,        // [hidden, kv_dim]  (W^T)
    o: *mut ggml_tensor,        // [hidden, hidden]  (W^T)
    gate: *mut ggml_tensor,     // [hidden, inter]   (W^T)
    up: *mut ggml_tensor,       // [hidden, inter]   (W^T)
    down: *mut ggml_tensor,     // [inter, hidden]   (W^T)
}

const ROPE_MODE: c_int = 2; // GGML_ROPE_TYPE_NEOX (HF "neox" / rotate_half)

/// Copy a HF weight `[out_dim, in_dim]` into a ggml tensor `[in_dim, out_dim]`
/// (i.e. store `W^T`) so that `ggml_mul_mat(w, x) == W @ x`.
unsafe fn weight_2d(
    ctx: *mut ggml_context,
    data: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> *mut ggml_tensor {
    assert_eq!(data.len(), out_dim * in_dim);
    let t = ggml_new_tensor_2d(ctx, ggml_type::F32, in_dim as i64, out_dim as i64);
    let dst = ggml_get_data_f32(t);
    // data is row-major HF W[o, i]; ggml wants W^T[i, j] = W[j, i].
    for i in 0..in_dim {
        for j in 0..out_dim {
            *dst.add(i * out_dim + j) = data[j * in_dim + i];
        }
    }
    t
}

unsafe fn weight_1d(ctx: *mut ggml_context, data: &[f32], dim: usize) -> *mut ggml_tensor {
    let t = ggml_new_tensor_1d(ctx, ggml_type::F32, dim as i64);
    let dst = ggml_get_data_f32(t);
    for i in 0..data.len() {
        *dst.add(i) = data[i];
    }
    t
}

impl Model {
    /// Build the model: allocate one ggml context holding all weight tensors
    /// (copied once, transposed, from the parsed safetensors) and reuse it
    /// across forwards.
    pub unsafe fn new(w: &SmolLmWeights) -> Self {
        let cfg = w.config;
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.kv_dim();

        let mem_size = 768 * 1024 * 1024; // 768 MiB: weights (~540) + activations
        let ctx = ggml_init(ggml_init_params {
            mem_size,
            mem_buffer: core::ptr::null_mut(),
            no_alloc: false,
        });
        assert!(!ctx.is_null(), "ggml_init failed for Model");

        let embed = weight_2d(ctx, &w.embed_tokens, vocab, hidden);
        let norm = weight_1d(ctx, &w.norm, hidden);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for lw in &w.layers {
            layers.push(LayerTensors {
                input_ln: weight_1d(ctx, &lw.input_layernorm, hidden),
                post_ln: weight_1d(ctx, &lw.post_attention_layernorm, hidden),
                q: weight_2d(ctx, &lw.q_proj, hidden, hidden),
                k: weight_2d(ctx, &lw.k_proj, kv_dim, hidden),
                v: weight_2d(ctx, &lw.v_proj, kv_dim, hidden),
                o: weight_2d(ctx, &lw.o_proj, hidden, hidden),
                gate: weight_2d(ctx, &lw.gate_proj, inter, hidden),
                up: weight_2d(ctx, &lw.up_proj, inter, hidden),
                down: weight_2d(ctx, &lw.down_proj, hidden, inter),
            });
        }

        Model {
            ctx,
            cfg,
            embed,
            norm,
            layers,
        }
    }

    /// Run a full-sequence forward and return the logits tensor `[vocab, seq]`.
    pub unsafe fn forward(&self, token_ids: &[i32]) -> *mut ggml_tensor {
        let cfg = self.cfg;
        let hidden = cfg.hidden_size as i64;
        let inter = cfg.intermediate_size as i64;
        let n_heads = cfg.num_attention_heads as i64;
        let n_kv = cfg.num_key_value_heads as i64;
        let head_dim = cfg.head_dim() as i64;
        let seq = token_ids.len() as i64;
        let ctx = self.ctx;
        let eps = cfg.rms_norm_eps;

        // Token-id and position tensors (int32).
        let tid = ggml_new_tensor_1d(ctx, ggml_type::I32, seq);
        {
            let p = ggml_get_data(tid) as *mut i32;
            for i in 0..token_ids.len() {
                *p.add(i) = token_ids[i];
            }
        }
        let pos = ggml_new_tensor_1d(ctx, ggml_type::I32, seq);
        {
            let p = ggml_get_data(pos) as *mut i32;
            for i in 0..token_ids.len() {
                *p.add(i) = i as i32;
            }
        }

        // Embedding lookup: [hidden, seq].
        let mut x = ggml_get_rows(ctx, self.embed, tid); // [hidden, seq]

        for layer in &self.layers {
            // --- Self-attention -------------------------------------------
            let xb = ggml_rms_norm(ctx, x, layer.input_ln, eps); // [hidden, seq]

            let q = ggml_reshape_3d(
                ctx,
                ggml_mul_mat(ctx, layer.q, xb),
                head_dim,
                n_heads,
                seq,
            ); // [head_dim, n_head, seq]
            let q = ggml_rope(ctx, q, pos, head_dim as c_int, ROPE_MODE);
            let q = ggml_reshape_3d(ctx, q, head_dim, seq, n_heads); // [head_dim, seq, n_head]

            let k = ggml_reshape_3d(
                ctx,
                ggml_mul_mat(ctx, layer.k, xb),
                head_dim,
                n_kv,
                seq,
            ); // [head_dim, n_kv, seq]
            let k = ggml_rope(ctx, k, pos, head_dim as c_int, ROPE_MODE);
            let k = ggml_reshape_3d(ctx, k, head_dim, seq, n_kv); // [head_dim, seq, n_kv]

            let v = ggml_reshape_3d(
                ctx,
                ggml_mul_mat(ctx, layer.v, xb),
                head_dim,
                n_kv,
                seq,
            ); // [head_dim, n_kv, seq]
            let v = ggml_reshape_3d(ctx, v, head_dim, seq, n_kv); // [head_dim, seq, n_kv]

            // GQA: repeat K/V heads to match the query head count.
            let k = ggml_repeat(ctx, k, q); // [head_dim, seq, n_head]
            let v = ggml_repeat(ctx, v, q); // [head_dim, seq, n_head]

            // scores = (k^T @ q) / sqrt(head_dim)  -> [seq, seq, n_head]
            let mut scores = ggml_mul_mat(ctx, k, q);
            scores = ggml_scale(ctx, scores, 1.0 / sqrtf(head_dim as f32));
            scores = ggml_soft_max(ctx, scores);

            // values = v^T @ scores  -> [head_dim, seq, n_head]
            let v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [seq, head_dim, n_head]
            let values = ggml_mul_mat(ctx, v_perm, scores); // [head_dim, seq, n_head]

            // back to [hidden, seq]
            let values = ggml_reshape_2d(ctx, values, hidden, seq); // [hidden, seq]
            let o = ggml_mul_mat(ctx, layer.o, values); // [hidden, seq]
            x = ggml_add(ctx, x, o);

            // --- MLP (SwiGLU) --------------------------------------------
            let xb = ggml_rms_norm(ctx, x, layer.post_ln, eps); // [hidden, seq]
            let gate = ggml_silu(ctx, ggml_mul_mat(ctx, layer.gate, xb)); // [inter, seq]
            let up = ggml_mul_mat(ctx, layer.up, xb); // [inter, seq]
            let act = ggml_mul(ctx, gate, up); // [inter, seq]
            let down = ggml_mul_mat(ctx, layer.down, act); // [hidden, seq]
            x = ggml_add(ctx, x, down);
        }

        // Final norm + tied LM head.
        x = ggml_rms_norm(ctx, x, self.norm, eps);
        // logits = embed^T @ x  ->  [vocab, seq]
        let logits = ggml_mul_mat(ctx, self.embed, x);
        logits
    }

    /// Compute the argmax of the logits at the last position (for greedy
    /// decoding). `seq` and `vocab` must match the last forward call.
    pub unsafe fn argmax_last(&self, logits: *mut ggml_tensor, seq: usize, vocab: usize) -> usize {
        let data = ggml_get_data_f32(logits);
        let base = data.add((seq - 1) * vocab);
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for i in 0..vocab {
            let v = *base.add(i);
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best
    }

    pub fn config(&self) -> SmolLmConfig {
        self.cfg
    }
}
