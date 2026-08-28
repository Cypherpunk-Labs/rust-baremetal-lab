#!/usr/bin/env python3
"""Independent numpy reference forward for SmolLM-135M (F32 safetensors).

Loads the same weights the Rust kernel embeds, runs a forward over the same
token ids, and writes logits as [seq, vocab] to build/ref_logits.bin for
comparison against the ggml forward (build/ggml_logits.bin).

NOTE: full (bidirectional) attention, no causal mask, to match the current
ggml graph exactly.
"""
import json
import struct
import numpy as np

MODEL = "../burn/model-builder/model.safetensors"
OUT = "build/ref_logits.bin"

# ---- config (matches SmolLmConfig::DEFAULT) ----
VOCAB, HIDDEN, INTER, LAYERS = 49152, 576, 1536, 30
N_HEADS, N_KV, MAX_POS, EPS, THETA = 9, 3, 2048, 1e-5, 10000.0
HEAD_DIM = HIDDEN // N_HEADS  # 64

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    tensors = {}
    for name, meta in header.items():
        if name.startswith("__"):
            continue
        dtype, shape, (s, e) = meta["dtype"], meta["shape"], meta["data_offsets"]
        buf = data[s:e]
        dt = {"F32": "<f4", "F16": "<f2"}[dtype]
        tensors[name] = np.frombuffer(buf, dtype=dt).reshape(shape).copy()
    return tensors

def rms_norm(x, w):
    # x: [..., HIDDEN]
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return x / r * w

def rope_neox(x, freqs_cos, freqs_sin):
    # x: [seq, ..., HEAD_DIM] ; freqs_cos/sin: [seq, HEAD_DIM//2]
    x1 = x[..., : HEAD_DIM // 2]
    x2 = x[..., HEAD_DIM // 2:]
    cos = freqs_cos[: x.shape[0]][:, None, :]
    sin = freqs_sin[: x.shape[0]][:, None, :]
    # NEOX rotate_half (matches ggml rotate_pairs): pair k uses cos[m,k], sin[m,k]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

def main():
    t = load_safetensors(MODEL)
    ids = [(i * 7) % VOCAB for i in range(16)]
    n_seq = len(ids)

    # precompute RoPE frequencies (NEOX: n_dims/2 pairs over head_dim)
    k = np.arange(HEAD_DIM // 2, dtype=np.float32)  # 0..31
    inv = THETA ** (-2.0 * k / HEAD_DIM)  # [HEAD_DIM//2]
    m = np.arange(n_seq, dtype=np.float32)[:, None]  # [seq, 1]
    ang = m * inv[None, :]  # [seq, HEAD_DIM//2]
    cos = np.cos(ang)  # [seq, HEAD_DIM//2]
    sin = np.sin(ang)

    x = t["model.embed_tokens.weight"][ids]  # [seq, HIDDEN]

    for li in range(LAYERS):
        wq = t[f"model.layers.{li}.self_attn.q_proj.weight"]
        wk = t[f"model.layers.{li}.self_attn.k_proj.weight"]
        wv = t[f"model.layers.{li}.self_attn.v_proj.weight"]
        wo = t[f"model.layers.{li}.self_attn.o_proj.weight"]
        ln1 = t[f"model.layers.{li}.input_layernorm.weight"]
        ln2 = t[f"model.layers.{li}.post_attention_layernorm.weight"]
        wg = t[f"model.layers.{li}.mlp.gate_proj.weight"]
        wu = t[f"model.layers.{li}.mlp.up_proj.weight"]
        wd = t[f"model.layers.{li}.mlp.down_proj.weight"]

        xb = rms_norm(x, ln1)
        q = xb @ wq.T  # [seq, HIDDEN]
        k = xb @ wk.T  # [seq, kv_dim]
        v = xb @ wv.T  # [seq, kv_dim]

        q = q.reshape(n_seq, N_HEADS, HEAD_DIM)
        k = k.reshape(n_seq, N_KV, HEAD_DIM)
        v = v.reshape(n_seq, N_KV, HEAD_DIM)

        q = rope_neox(q, cos, sin)
        k = rope_neox(k, cos, sin)

        out = np.zeros((n_seq, N_HEADS, HEAD_DIM), dtype=np.float32)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        for h in range(N_HEADS):
            kvh = h * N_KV // N_HEADS
            qh, kh, vh = q[:, h], k[:, kvh], v[:, kvh]  # [seq, HEAD_DIM]
            scores = qh @ kh.T * scale  # [seq, seq]
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            scores /= scores.sum(axis=-1, keepdims=True)
            out[:, h] = scores @ vh
        attn = out.reshape(n_seq, HIDDEN)
        attn = attn @ wo.T
        x = x + attn

        xb = rms_norm(x, ln2)
        gate = xb @ wg.T
        gate = gate / (1.0 + np.exp(-gate))  # silu
        up = xb @ wu.T
        act = gate * up
        down = act @ wd.T
        x = x + down

    x = rms_norm(x, t["model.norm.weight"])
    logits = x @ t["model.embed_tokens.weight"].T  # tied LM head: [seq, VOCAB]

    with open(OUT, "wb") as f:
        f.write(struct.pack("<ii", n_seq, VOCAB))
        f.write(logits.astype("<f4").tobytes())
    print("wrote", OUT, "logits", logits.shape, "finite", np.isfinite(logits).all())

if __name__ == "__main__":
    main()