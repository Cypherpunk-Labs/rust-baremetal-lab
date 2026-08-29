//! Pure-Rust reference forward for SmolLM-135M (F32), used as the oracle
//! against which ggml output is compared.
//!
//! Layout conventions (canonical, matching the ggml `layer0_diag` dump):
//!   * weights   : HF row-major `[out, in]`  ->  linear(x) = x @ W^T.
//!   * activations: `[dim, seq]` column-major (matches ggml `ne0 = dim`).
//!   * q/k/v, values: `[head_dim, seq, n_head]` flat = d + s*head_dim + h*head_dim*seq
//!     (d fastest, then token s, then head h).
//!   * scores    : `[seq_key, seq_query, n_head]` flat = sk + sq*seq + h*seq*seq.
//!
//! GQA is **grouped** (`kv head = query_head / q_per_kv`) and attention is
//! **full / non-causal** — both chosen to match the ggml graph under test.

use model::SmolLmConfig;
use model::SmolLmWeights;

pub struct Ref {
    pub cfg: SmolLmConfig,
    pub w: SmolLmWeights,
}

const EPS: f32 = 1e-5;

impl Ref {
    pub fn new(w: SmolLmWeights) -> Ref {
        Ref { cfg: w.config, w }
    }

    fn rms_norm(&self, x: &[f32], dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for (row, o) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv = 1.0 / (mean_sq + EPS).sqrt();
            for (v, ov) in row.iter().zip(o.iter_mut()) {
                *ov = v * inv;
            }
        }
        out
    }

    fn affine(&self, x: &[f32], w: &[f32], _dim: usize) -> Vec<f32> {
        // x is rows of `dim`; w is length `dim`; multiply per-element.
        x.iter().zip(w.iter().cycle()).map(|(a, b)| a * b).collect()
    }

    /// y[o, s] = sum_i W[o*in + i] * x[i, s]. Column-major `[dim, seq]`:
    /// element (d, s) at `d + s*dim`. Weights are HF row-major `[out, in]`.
    fn linear(&self, x: &[f32], w: &[f32], in_dim: usize, out_dim: usize, seq: usize) -> Vec<f32> {
        // x: [in_dim, seq] flat i + s*in_dim; y: [out_dim, seq] flat o + s*out_dim
        let mut y = vec![0.0f32; out_dim * seq];
        for o in 0..out_dim {
            for i in 0..in_dim {
                let wv = w[o * in_dim + i];
                for s in 0..seq {
                    y[o + s * out_dim] += wv * x[i + s * in_dim];
                }
            }
        }
        y
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|v| *v / (1.0 + (-*v).exp())).collect()
    }

    /// cos & sin for an angle (f32).
    fn cossin(&self, ang: f32) -> (f32, f32) {
        (ang.cos(), ang.sin())
    }

    /// RoPE (NEOX / rotate_half) on `[head_dim, seq, n_head]` (flat
    /// `d + s*head_dim + h*head_dim*seq`). Pair k uses (d=k, d=k+half).
    pub fn rope_neox(&self, mut q: Vec<f32>, head_dim: usize, n_head: usize, seq: usize, positions: &[i32]) -> Vec<f32> {
        let half = head_dim / 2;
        for s in 0..seq {
            let m = positions[s] as f32;
            for h in 0..n_head {
                for k in 0..half {
                    let ang = m * self.cfg.rope_theta.powf(-2.0 * k as f32 / head_dim as f32);
                    let (c, sn) = self.cossin(ang);
                    let i0 = k + s * head_dim + h * head_dim * seq;
                    let i1 = (k + half) + s * head_dim + h * head_dim * seq;
                    let x0 = q[i0];
                    let x1 = q[i1];
                    q[i0] = x0 * c - x1 * sn;
                    q[i1] = x0 * sn + x1 * c;
                }
            }
        }
        q
    }

    /// Full forward. Returns (per-layer intermediates for layer 0, final logits).
    /// logits: [vocab, seq].
    pub fn forward(&self, token_ids: &[i32]) -> (Layer0Diag, Vec<f32>) {
        let cfg = self.cfg;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.kv_dim();
        let seq = token_ids.len();
        let q_per_kv = n_heads / n_kv; // 3

        // embed: [hidden, seq]
        let mut x: Vec<f32> = Vec::with_capacity(h * seq);
        for s in 0..seq {
            let tok = token_ids[s] as usize;
            for d in 0..h {
                x.push(self.w.embed_tokens[tok * h + d]);
            }
        }

        let mut diag: Option<Layer0Diag> = None;

        for (li, lw) in self.w.layers.iter().enumerate() {
            // rms_norm -> affine
            let rn = self.rms_norm(&x, h);
            let xb = self.affine(&rn, &lw.input_layernorm, h); // [h, seq]

            // q/k/v projections -> [out_dim, seq]
            let q = self.linear(&xb, &lw.q_proj, h, h, seq); // [n_heads*head_dim, seq]
            let k = self.linear(&xb, &lw.k_proj, h, kv_dim, seq); // [n_kv*head_dim, seq]
            let v = self.linear(&xb, &lw.v_proj, h, kv_dim, seq);

            // reshape q to [head_dim, n_head, seq]; k,v to [head_dim, n_kv, seq]
            let q3 = self.to_dhs(&q, head_dim, n_heads, seq); // [head_dim, n_head, seq]
            let k3 = self.to_dhs(&k, head_dim, n_kv, seq);
            let v3 = self.to_dhs(&v, head_dim, n_kv, seq);

            let positions: Vec<i32> = (0..seq as i32).collect();
            let q3_for_diag = q3.clone();
            let qr = self.rope_neox(q3, head_dim, n_heads, seq, &positions);
            let kr = self.rope_neox(k3, head_dim, n_kv, seq, &positions);
            let vr = v3;

            // attention. For each query head hg, kv head = hg / q_per_kv (GROUPED).
            // scores: [seq_key, seq_query, n_head] flat sk + sq*seq + hg*seq*seq
            // values: [head_dim, seq, n_head]      flat d + s*head_dim + hg*head_dim*seq
            let mut values_out = vec![0.0f32; head_dim * n_heads * seq];
            let mut scores_all = vec![0.0f32; seq * seq * n_heads];
            for hg in 0..n_heads {
                let kh = hg / q_per_kv;
                for sc in 0..seq {
                    let mut maxv = f32::NEG_INFINITY;
                    let mut raw = vec![0.0f32; seq];
                    for sr in 0..seq {
                        if sr > sc {
                            continue; // causal mask: query attends only to keys <= itself
                        }
                        let mut acc = 0.0f32;
                        for d in 0..head_dim {
                            acc += qr[d + sc * head_dim + hg * head_dim * seq]
                                * kr[d + sr * head_dim + kh * head_dim * seq];
                        }
                        let s = acc / (head_dim as f32).sqrt();
                        raw[sr] = s;
                        if s > maxv {
                            maxv = s;
                        }
                    }
                    let mut sum = 0.0f32;
                    for sr in 0..seq {
                        if sr > sc {
                            continue;
                        }
                        let e = (raw[sr] - maxv).exp();
                        raw[sr] = e;
                        sum += e;
                    }
                    for sr in 0..seq {
                        if sr > sc {
                            continue;
                        }
                        let p = raw[sr] / sum;
                        scores_all[sr + sc * seq + hg * seq * seq] = p;
                        for d in 0..head_dim {
                            values_out[d + sc * head_dim + hg * head_dim * seq] +=
                                p * vr[d + sr * head_dim + kh * head_dim * seq];
                        }
                    }
                }
            }

            // o = Wo @ values. `values_out` is stored [head_dim, seq, n_head]
            // (flat d + s*head_dim + h*head_dim*seq) for diagnostics; the linear
            // needs [hidden, seq] flat (h*head_dim + d) + s*hidden — a TRUE
            // transpose, not a reshape. Remap into o_order before the matmul.
            let mut values_hs = vec![0.0f32; head_dim * n_heads * seq];
            for hg in 0..n_heads {
                for sc in 0..seq {
                    for d in 0..head_dim {
                        values_hs[(hg * head_dim + d) + sc * (head_dim * n_heads)] =
                            values_out[d + sc * head_dim + hg * head_dim * seq];
                    }
                }
            }
            let o = self.linear(&values_hs, &lw.o_proj, h, h, seq);
            for i in 0..x.len() {
                x[i] += o[i];
            }
            // if li == 0 capture diag before MLP

            // MLP
            let rn2 = self.rms_norm(&x, h);
            let xb2 = self.affine(&rn2, &lw.post_attention_layernorm, h);
            let gate = self.silu(&self.linear(&xb2, &lw.gate_proj, h, inter, seq));
            let up = self.linear(&xb2, &lw.up_proj, h, inter, seq);
            let act: Vec<f32> = gate.iter().zip(up.iter()).map(|(a, b)| a * b).collect();
            let down = self.linear(&act, &lw.down_proj, inter, h, seq);
            for i in 0..x.len() {
                x[i] += down[i];
            }

            if li == 0 {
                diag = Some(Layer0Diag {
                    xb,     // [h, seq] post input rms_norm
                    q: q3_for_diag, // [head_dim, seq, n_head] pre-rope
                    qr,     // [head_dim, seq, n_head] post-rope
                    kr,     // [head_dim, seq, n_kv] post-rope
                    vr,     // [head_dim, seq, n_kv]
                    scores: scores_all, // [seq, seq, n_head]
                    values: values_out, // [head_dim, seq, n_head]
                    o,      // [h, seq]
                    x: x.clone(), // post-attn+MLP (end of layer 0)
                });
            }
        }

        // final rms_norm -> affine -> logits [vocab, seq]
        let rn = self.rms_norm(&x, h);
        let xf = self.affine(&rn, &self.w.norm, h);
        let logits = self.linear(&xf, &self.w.embed_tokens, h, vocab, seq);

        (diag.unwrap(), logits)
    }

    /// Reshape a `[out_dim, seq]` linear output into `[head_dim, seq, n_head]`
    /// (flat `d + s*head_dim + h*head_dim*seq`). Row `h*head_dim + d`.
    fn to_dhs(&self, lin: &[f32], head_dim: usize, n_head: usize, seq: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; head_dim * n_head * seq];
        for h in 0..n_head {
            for d in 0..head_dim {
                let lin_row = h * head_dim + d;
                for s in 0..seq {
                    out[d + s * head_dim + h * head_dim * seq] =
                        lin[lin_row + s * (head_dim * n_head)];
                }
            }
        }
        out
    }
}

/// Layer-0 diagnostics, all in documented layouts.
pub struct Layer0Diag {
    pub xb: Vec<f32>,      // [hidden, seq] post input rms_norm
    pub q: Vec<f32>,       // [head_dim, seq, n_head] pre-rope
    pub qr: Vec<f32>,      // [head_dim, seq, n_head] post-rope
    pub kr: Vec<f32>,      // [head_dim, seq, n_kv] post-rope
    pub vr: Vec<f32>,      // [head_dim, seq, n_kv]
    pub scores: Vec<f32>,  // [seq_key, seq_query, n_head]
    pub values: Vec<f32>,  // [head_dim, seq, n_head]
    pub o: Vec<f32>,       // [hidden, seq] attention out
    pub x: Vec<f32>,       // [hidden, seq] end of layer 0 (post MLP)
}
