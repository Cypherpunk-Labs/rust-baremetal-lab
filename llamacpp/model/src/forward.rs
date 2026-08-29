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
use alloc::vec;
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

/// Expand K/V from [head_dim, seq, n_kv] to grouped [head_dim, seq, n_head].
/// SmolLM uses **grouped** GQA: query head `h` attends to kv head `h // rep`.
/// `ggml_repeat` would interleave (`h % n_kv`), which is wrong for this model,
/// so we gather each kv head `rep` times contiguously.
/// `idx` is an `[n_head]` i32 tensor with `idx[p] = p / rep`.
unsafe fn gqa_group_expand(
    ctx: *mut ggml_context,
    src: *mut ggml_tensor, // [head_dim, seq, n_kv]
    idx: *mut ggml_tensor, // [n_head] i32, idx[p] = p / rep
    n_kv: i64,
    n_head: i64,
    head_dim: i64,
    seq: i64,
) -> *mut ggml_tensor {
    // ggml_get_rows gathers along ne[1] (rows), keeping ne[0] (row length) as
    // the leading dim. Reshape so each kv head is one ROW of full
    // [head_dim*seq] data: -> [head_dim*seq, n_kv]. Then gather grouped heads.
    let a = ggml_reshape_2d(ctx, src, head_dim * seq, n_kv); // [head_dim*seq, n_kv]
    let rows = ggml_get_rows(ctx, a, idx); // [head_dim*seq, n_head]
    ggml_reshape_3d(ctx, rows, head_dim, seq, n_head) // [head_dim, seq, n_head]
}

/// Copy a HF weight `[out_dim, in_dim]` into a ggml tensor `[ne0, ne1]`.
/// Empirically ggml_mul_mat(w, x) computes `W @ x` when the weight bytes are
/// stored verbatim (HF row-major `[out, in]`) with ne0 = in, ne1 = out. (The
/// earlier all-ones harness test could not distinguish the orientation; the
/// q-projection parity test showed no transpose is needed.)
unsafe fn weight_2d(
    ctx: *mut ggml_context,
    data: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> *mut ggml_tensor {
    assert_eq!(data.len(), out_dim * in_dim);
    let t = ggml_new_tensor_2d(ctx, ggml_type::F32, in_dim as i64, out_dim as i64);
    let dst = ggml_get_data_f32(t);
    for i in 0..data.len() {
        *dst.add(i) = data[i];
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

/// RMSNorm(a) * weight: ggml applies the affine weight via a separate multiply.
unsafe fn rms_norm(
    ctx: *mut ggml_context,
    x: *mut ggml_tensor,
    weight: *mut ggml_tensor,
    eps: f32,
) -> *mut ggml_tensor {
    ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight)
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
    /// Run the full forward pass and return the logits (`[vocab, seq]` flat)
    /// for every position. The graph is built in a FRESH context that
    /// references the persistent weight tensors, then freed — so this is safe
    /// to call repeatedly in a generation loop (no context growth / leak).
    pub unsafe fn forward(&self, token_ids: &[i32]) -> Vec<f32> {
        let cfg = self.cfg;
        let hidden = cfg.hidden_size as i64;
        let n_heads = cfg.num_attention_heads as i64;
        let n_kv = cfg.num_key_value_heads as i64;
        let head_dim = cfg.head_dim() as i64;
        let seq = token_ids.len() as i64;
        // Graph-only context (intermediates). Weights stay in `self.ctx`.
        let ctx = ggml_init(ggml_init_params {
            mem_size: 256 * 1024 * 1024,
            mem_buffer: core::ptr::null_mut(),
            no_alloc: false,
        });
        assert!(!ctx.is_null(), "forward: graph context init failed");
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

        // Grouped-GQA gather index: idx[p] = p / rep, rep = n_heads / n_kv.
        let rep = n_heads / n_kv;
        let idx = ggml_new_tensor_1d(ctx, ggml_type::I32, n_heads);
        {
            let p = ggml_get_data(idx) as *mut i32;
            for hh in 0..n_heads {
                *p.add(hh as usize) = (hh / rep) as i32;
            }
        }

        // Embedding lookup: [hidden, seq].
        let mut x = ggml_get_rows(ctx, self.embed, tid); // [hidden, seq]

        for layer in &self.layers {
            // --- Self-attention -------------------------------------------
            let xb = rms_norm(ctx, x, layer.input_ln, eps); // [hidden, seq]

            // q/k/v come out of mul_mat as [hidden, seq]; reshape to
            // [hd, n_head, seq], apply RoPE, then PERMUTE to the true
            // [hd, seq, n_head] canonical order (a plain reshape would scramble
            // the axes since reshape does not relayout memory).
            let q = ggml_reshape_3d(
                ctx,
                ggml_mul_mat(ctx, layer.q, xb),
                head_dim,
                n_heads,
                seq,
            ); // [head_dim, n_head, seq]
            let q = ggml_cont(
                ctx,
                ggml_permute(
                    ctx,
                    ggml_rope(ctx, q, pos, head_dim as c_int, ROPE_MODE),
                    0,
                    2,
                    1,
                    3,
                ),
            ); // [head_dim, seq, n_head]

            let k = ggml_reshape_3d(
                ctx,
                ggml_mul_mat(ctx, layer.k, xb),
                head_dim,
                n_kv,
                seq,
            ); // [head_dim, n_kv, seq]
            let k = ggml_cont(
                ctx,
                ggml_permute(
                    ctx,
                    ggml_rope(ctx, k, pos, head_dim as c_int, ROPE_MODE),
                    0,
                    2,
                    1,
                    3,
                ),
            ); // [head_dim, seq, n_kv]

            let v = ggml_cont(
                ctx,
                ggml_permute(
                    ctx,
                    ggml_reshape_3d(
                        ctx,
                        ggml_mul_mat(ctx, layer.v, xb),
                        head_dim,
                        n_kv,
                        seq,
                    ),
                    0,
                    2,
                    1,
                    3,
                ),
            ); // [head_dim, seq, n_kv]

            // GQA: expand K/V heads to grouped order (SmolLM: h // rep), not
            // ggml_repeat's interleaved mapping.
            let k = gqa_group_expand(ctx, k, idx, n_kv, n_heads, head_dim, seq); // [head_dim, seq, n_head]
            let v = gqa_group_expand(ctx, v, idx, n_kv, n_heads, head_dim, seq); // [head_dim, seq, n_head]

            // scores = (k^T @ q) / sqrt(head_dim)  -> [ne0=key, ne1=query, n_head]
            // ggml_soft_max softmaxes over ne0 (the KEY axis) = correct attention.
            let mut scores = ggml_mul_mat(ctx, k, q);
            scores = ggml_scale(ctx, scores, 1.0 / sqrtf(head_dim as f32));
            // Causal mask: zero scores where key index > query index (ne1 >= n_past + ne0).
            scores = ggml_diag_mask_inf(ctx, scores, 0);
            scores = ggml_soft_max(ctx, scores);

            // values = v^T @ scores  -> [head_dim, seq, n_head]
            let v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [seq, head_dim, n_head]
            let values = ggml_mul_mat(ctx, v_perm, scores); // [head_dim, seq, n_head]

            // Merge heads into the hidden dim with a TRUE transpose: [hd, seq,
            // n_head] -> [hd, n_head, seq] so that flat (hidden, seq) reads
            // (h*hd + d) + s*hidden. A plain reshape would NOT relayout and
            // would scramble the axes (the bug that broke attention).
            let values = ggml_reshape_2d(
                ctx,
                ggml_cont(ctx, ggml_permute(ctx, values, 0, 2, 1, 3)), // [hd, n_head, seq]
                hidden,
                seq,
            ); // [hidden, seq]
            let o = ggml_mul_mat(ctx, layer.o, values); // [hidden, seq]
            x = ggml_add(ctx, x, o);

            // --- MLP (SwiGLU) --------------------------------------------
            let xb = rms_norm(ctx, x, layer.post_ln, eps); // [hidden, seq]
            let gate = ggml_silu(ctx, ggml_mul_mat(ctx, layer.gate, xb)); // [inter, seq]
            let up = ggml_mul_mat(ctx, layer.up, xb); // [inter, seq]
            let act = ggml_mul(ctx, gate, up); // [inter, seq]
            let down = ggml_mul_mat(ctx, layer.down, act); // [hidden, seq]
            x = ggml_add(ctx, x, down);
        }

        // Final norm + tied LM head.
        x = rms_norm(ctx, x, self.norm, eps);
        // logits = embed^T @ x  ->  [vocab, seq]
        let logits = ggml_mul_mat(ctx, self.embed, x);

        // Build + execute the graph. Use a generous node capacity for the
        // 30-layer graph (default 2048 is insufficient).
        let gf = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(gf, logits);
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        let n = (seq as usize) * (cfg.vocab_size as usize);
        let data = ggml_get_data_f32(logits);
        let out: Vec<f32> = (0..n).map(|i| *data.add(i)).collect();
        ggml_free(ctx);
        out
    }

    /// Greedy-decode the next token from logits returned by `forward`
    /// (`[vocab, seq]` flat). Returns the argmax at the LAST position.
    pub fn argmax_last(logits: &[f32], seq: usize, vocab: usize) -> usize {
        let base = (seq - 1) * vocab;
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for i in 0..vocab {
            let v = logits[base + i];
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


    /// Diagnostic: extract layer-0 attention intermediates, all returned in a
    /// single documented logical layout (so callers never re-derive strides).
    ///
    /// Layouts (same as `parity::Ref::Layer0Diag`):
    ///   xb     : [hidden, seq]             flat = d + s*hidden
    ///   q_pre  : [head_dim, seq, n_head]   flat = d + s*head_dim + h*head_dim*seq
    ///   q_post : [head_dim, seq, n_head]
    ///   k_post : [head_dim, seq, n_kv]
    ///   v_     : [head_dim, seq, n_kv]
    ///   scores : [seq, seq, n_head]        flat = sk + sq*seq + h*seq*seq
    ///   values : [head_dim, seq, n_head]
    ///   o      : [hidden, seq]
    pub unsafe fn layer0_diag(&self, t0: i32, t1: i32) -> Option<Layer0Diag> {
        self.layer0_diag_full(&[t0, t1])
    }

    /// Diagnostic: extract layer-0 attention intermediates for a prompt of any
    /// length, all in the documented logical layout (see `layer0_diag`).
    pub unsafe fn layer0_diag_full(&self, ids: &[i32]) -> Option<Layer0Diag> {
        if self.layers.is_empty() {
            return None;
        }
        let ctx = self.ctx;
        let cfg = self.cfg;
        let hidden = cfg.hidden_size as i64;
        let n_heads = cfg.num_attention_heads as i64;
        let n_kv = cfg.num_key_value_heads as i64;
        let head_dim = cfg.head_dim() as i64;
        let seq = ids.len() as i64;
        let eps = cfg.rms_norm_eps;

        let tid = ggml_new_tensor_1d(ctx, ggml_type::I32, seq);
        {
            let p = ggml_get_data(tid) as *mut i32;
            for i in 0..ids.len() {
                *p.add(i) = ids[i];
            }
        }
        let pos = ggml_new_tensor_1d(ctx, ggml_type::I32, seq);
        {
            let p = ggml_get_data(pos) as *mut i32;
            for i in 0..ids.len() {
                *p.add(i) = i as i32;
            }
        }
        let rep = n_heads / n_kv;
        let idx = ggml_new_tensor_1d(ctx, ggml_type::I32, n_heads);
        {
            let p = ggml_get_data(idx) as *mut i32;
            for hh in 0..n_heads {
                *p.add(hh as usize) = (hh / rep) as i32;
            }
        }
        let layer = &self.layers[0];
        let x = ggml_get_rows(ctx, self.embed, tid); // [hidden, seq]
        let xb = rms_norm(ctx, x, layer.input_ln, eps); // [hidden, seq]

        // q/k/v come out of mul_mat as [hidden, seq]; reshape to [hd, n_head, seq],
        // apply RoPE in that layout (seq is axis 2), then PERMUTE to the true
        // [hd, seq, n_head] canonical order. A plain reshape would scramble the
        // axes (reshape does not relayout memory), so permute is required.
        let q_pre = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer.q, xb), head_dim, n_heads, seq);
        let q_post = ggml_cont(
            ctx,
            ggml_permute(
                ctx,
                ggml_rope(ctx, q_pre, pos, head_dim as c_int, ROPE_MODE),
                0,
                2,
                1,
                3,
            ),
        ); // [head_dim, seq, n_head]

        let k_pre = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer.k, xb), head_dim, n_kv, seq);
        let k_post = ggml_cont(
            ctx,
            ggml_permute(
                ctx,
                ggml_rope(ctx, k_pre, pos, head_dim as c_int, ROPE_MODE),
                0,
                2,
                1,
                3,
            ),
        ); // [head_dim, seq, n_kv]
        let v3 = ggml_cont(
            ctx,
            ggml_permute(
                ctx,
                ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer.v, xb), head_dim, n_kv, seq),
                0,
                2,
                1,
                3,
            ),
        ); // [head_dim, seq, n_kv]

        let k_r = gqa_group_expand(ctx, k_post, idx, n_kv, n_heads, head_dim, seq); // [head_dim, seq, n_head]
        let v_r = gqa_group_expand(ctx, v3, idx, n_kv, n_heads, head_dim, seq); // [head_dim, seq, n_head]
        // scores = K @ Q^T. ggml's mul_mat gives result [ne0=key, ne1=query,
        // ne2=head] and ggml_soft_max softmaxes over ne0 (the KEY axis) — which
        // is exactly what scaled-dot-product attention needs. (Q @ K^T would
        // put the query on ne0 and softmax over the wrong axis.)
        let mut scores = ggml_mul_mat(ctx, k_r, q_post); // [seq_k, seq_q, n_head]
        scores = ggml_scale(ctx, scores, 1.0 / sqrtf(head_dim as f32));
        scores = ggml_diag_mask_inf(ctx, scores, 0); // causal mask
        scores = ggml_soft_max(ctx, scores); // softmax over key axis
        let v_perm = ggml_cont(ctx, ggml_permute(ctx, v_r, 1, 0, 2, 3)); // [seq_k, head_dim, n_head]
        let values = ggml_mul_mat(ctx, v_perm, scores); // -> [head_dim, seq_q, n_head]
        // True transpose so o = Wo @ values reads [hidden, seq] correctly.
        let values2d = ggml_reshape_2d(
            ctx,
            ggml_cont(ctx, ggml_permute(ctx, values, 0, 2, 1, 3)), // [hd, n_head, seq]
            hidden,
            seq,
        ); // [hidden, seq]
        let o = ggml_mul_mat(ctx, layer.o, values2d); // [hidden, seq]

        // Dump raw buffers. IMPORTANT: ggml reshape/permute do NOT relayout
        // memory, so each tensor's buffer is in the order of the op that
        // produced it. Read with the SOURCE-order index, not a naive reshape.
        //   * q_pre/q_post: mul_mat output [hd*nhead, seq]  -> buf[(h*hd+d) + s*(hd*nh)]
        //   * k_post/v3   : [hd*nkv,   seq]                 -> buf[(h*hd+d) + s*(hd*nk)]
        //   * k_r/values  : contiguous [hd*seq, nhead]      -> buf[(d+s*hd) + h*(hd*seq)]
        //   * scores/xb/o : contiguous in canonical layout
        let gf = ggml_new_graph_custom(ctx, 4096, false);
        for t in [o, scores, q_post, k_post, k_r, v3, values2d, xb] {
            ggml_build_forward_expand(gf, t);
        }
        ggml_graph_compute_with_ctx(ctx, gf, 1);

        let hd = head_dim as usize;
        let nh = n_heads as usize;
        let nk = n_kv as usize;
        let se = seq as usize;
        let hi = hidden as usize;

        // remap a [hd*n_h, seq]-order buffer into [head_dim, seq, n_h] (canonical).
        let remap_headseq = |buf: &[f32], n_h: usize, n_dim: usize, stride: usize| -> Vec<f32> {
            let mut out = vec![0f32; n_dim * n_h * se];
            for h in 0..n_h {
                for d in 0..n_dim {
                    for s in 0..se {
                        out[d + s * n_dim + h * n_dim * se] = buf[(h * n_dim + d) + s * stride];
                    }
                }
            }
            out
        };

        let xb_v = copy_flat(ggml_get_data_f32(xb), hi * se); // [hidden, seq]
        let q_pre_v = remap_headseq(&copy_flat(ggml_get_data_f32(q_pre), hi * se), nh, hd, hd * nh);
        // q_post/k_post/v3 are now materialized in true [hd, seq, n_head] order.
        let q_post_v = copy_flat(ggml_get_data_f32(q_post), hi * se); // [hd, seq, nh]
        let k_post_v = copy_flat(ggml_get_data_f32(k_post), nk * hd * se); // [hd, seq, nk]
        let kr_v = copy_flat(ggml_get_data_f32(k_r), nh * hd * se); // [hd, seq, nh]
        let v_v = copy_flat(ggml_get_data_f32(v3), nk * hd * se); // [hd, seq, nk]
        // scores = mul_mat result [ne0=key(sk), ne1=query(sq), ne2=head]; the
        // buffer index is sk + sq*seq + h*seq*seq. Reorder into the reference
        // layout [sq*seq + sk + h*seq*seq] so callers read [query, key].
        let raw_scores = copy_flat(ggml_get_data_f32(scores), se * se * nh);
        let mut scores_v = vec![0f32; se * se * nh];
        for h in 0..nh {
            for sq in 0..se {
                for sk in 0..se {
                    scores_v[sq * se + sk + h * se * se] = raw_scores[sk + sq * se + h * se * se];
                }
            }
        }
        // values2d is now a true [hidden, seq] (hd*nh, seq) materialization.
        // Remap into the canonical [head_dim, seq, n_head] diag layout.
        let raw_values = copy_flat(ggml_get_data_f32(values2d), hd * nh * se);
        let mut values_v = vec![0f32; hd * nh * se];
        for h in 0..nh {
            for s in 0..se {
                for d in 0..hd {
                    values_v[d + s * hd + h * hd * se] = raw_values[(h * hd + d) + s * (hd * nh)];
                }
            }
        }
        let o_buf = copy_flat(ggml_get_data_f32(o), hi * se);

        Some(Layer0Diag {
            xb: xb_v,
            q_pre: q_pre_v,
            q_post: q_post_v,
            k_post: k_post_v,
            k_repeat: kr_v,
            v_: v_v,
            scores: scores_v,
            values: values_v,
            o: o_buf,
        })
    }
}

unsafe fn copy_flat(p: *const f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| *p.add(i)).collect()
}

/// Layer-0 attention intermediates with fully documented logical layouts
/// (see `Model::layer0_diag`). All element-wise comparable to
/// `parity::Ref::Layer0Diag`.
pub struct Layer0Diag {
    pub xb: Vec<f32>,
    pub q_pre: Vec<f32>,
    pub q_post: Vec<f32>,
    pub k_post: Vec<f32>,
    pub k_repeat: Vec<f32>,
    pub v_: Vec<f32>,
    pub scores: Vec<f32>,
    pub values: Vec<f32>,
    pub o: Vec<f32>,
}
