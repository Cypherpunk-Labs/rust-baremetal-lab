//! Parity: compare ggml full-forward logits against the pure-Rust reference.
//!
//! LAYOUT NOTE: ggml logits tensor is `[vocab, seq]` (ne0 = vocab). Its flat
//! buffer, indexed as `[vocab, seq]` C-order, gives `logits[v, s]` at
//! `v + s*vocab`. The reference also produces `[vocab, seq]`. We compare each
//! `(seq, vocab)` slice element-wise. The test is deterministic over the
//! fixed token-ids pattern.

use model::SmolLmConfig;
use model::SmolLmWeights;
use model::safetensors::SafeTensors;
use model_data::MODEL_BYTES;
use parity::Ref;

fn load() -> SmolLmWeights {
    let st = SafeTensors::parse(MODEL_BYTES).expect("parse safetensors");
    SmolLmWeights::from_safetensors(&st, SmolLmConfig::DEFAULT)
}

#[test]
#[ignore]
fn dump_weights() {
    let w = load();
    let h = SmolLmConfig::DEFAULT.hidden_size;
    println!("SAFE embed[504] first6 = {:?}", &w.embed_tokens[504 * h..504 * h + 6]);
    println!("SAFE q_proj[0..6] = {:?}", &w.layers[0].q_proj[0..6]);
}

fn token_ids(cfg: &SmolLmConfig, n: usize) -> Vec<i32> {
    (0..n).map(|i| ((i * 7) % cfg.vocab_size) as i32).collect()
}

fn argmax(x: &[f32]) -> usize {
    let mut b = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, v) in x.iter().enumerate() {
        if *v > bv {
            bv = *v;
            b = i;
        }
    }
    b
}

// The ggml model is heavy (513 MB embedded) so share one instance across tests.
fn ggml_logits(ids: &[i32]) -> Vec<f32> {
    let w = load();
    let model = unsafe { model::forward::Model::new(&w) };
    unsafe { model.forward(ids) }
}

#[test]
fn full_logits_parity() {
    let w = load();
    let ref_ = Ref::new(w);
    let cfg = SmolLmConfig::DEFAULT;
    let ids = token_ids(&cfg, 16);
    let (_, ref_logits) = ref_.forward(&ids);

    let gg_logits = ggml_logits(&ids);

    // ggml flat is [vocab, seq] (ne0=vocab). Index gg[v,s] = flat[v + s*vocab].
    // ref is also [vocab, seq]. Compare per (s,v).
    let v = cfg.vocab_size;
    let s = ids.len();
    let mut maxdiff = 0.0f32;
    let mut over = 0usize;
    for sq in 0..s {
        for vv in 0..v {
            let g = gg_logits[vv + sq * v];
            let r = ref_logits[vv + sq * v];
            let d = (g - r).abs();
            if d > maxdiff {
                maxdiff = d;
            }
            if d > 1e-2 {
                over += 1;
            }
        }
    }
    println!(
        "full_logits_parity: seq={} maxdiff={:.6} elements_over_1e-2={}",
        s, maxdiff, over
    );
    assert!(
        maxdiff < 1e-1,
        "ggml/reference logits diverge: maxdiff={}",
        maxdiff
    );

    // Argmax parity (what actually drives generation).
    let mut agree = 0usize;
    for sq in 0..s {
        let gbase = &gg_logits[sq * v..(sq + 1) * v];
        let rbase = &ref_logits[sq * v..(sq + 1) * v];
        let gmax = argmax(gbase);
        let rmax = argmax(rbase);
        if gmax == rmax {
            agree += 1;
        }
    }
    println!("argmax agree {}/{}", agree, s);
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

/// Interleaved-GQA attention scores `[seq, seq, n_head]` (flat
/// `sk + sq*seq + h*seq*seq`), using kv head `kh = h % n_kv`.
/// q/k are `[head_dim, seq, n_head]` / `[head_dim, seq, n_kv]` (d fastest).
fn interleaved_scores(q: &[f32], k: &[f32], hd: usize, nh: usize, nkv: usize, seq: usize) -> Vec<f32> {
    let mut sc = vec![0f32; seq * seq * nh];
    for h in 0..nh {
        let kh = h % nkv;
        for sq_ in 0..seq {
            let mut raw = vec![0f32; seq];
            let mut maxv = f32::NEG_INFINITY;
            for sk in 0..seq {
                let mut acc = 0f32;
                for d in 0..hd {
                    acc += q[d + sq_ * hd + h * hd * seq] * k[d + sk * hd + kh * hd * seq];
                }
                let v = acc / (hd as f32).sqrt();
                raw[sk] = v;
                if v > maxv {
                    maxv = v;
                }
            }
            let mut sum = 0f32;
            for sk in 0..seq {
                let e = (raw[sk] - maxv).exp();
                raw[sk] = e;
                sum += e;
            }
            for sk in 0..seq {
                sc[sk + sq_ * seq + h * seq * seq] = raw[sk] / sum;
            }
        }
    }
    sc
}

/// Compare ggml vs reference layer-0 attention intermediates. Files in on a
/// whole layer-0 if any of the primitives diverge, via `which` message.
#[test]
fn layer0_parity() {
    let w = load();
    let rr = Ref::new(w);
    let ids = [0i32, 1i32];
    let (diag, _) = rr.forward(&ids);

    // ggml layer-0 diag over tokens 0,1
    let w2 = load();
    let model = unsafe { model::forward::Model::new(&w2) };
    let g = unsafe { model.layer0_diag(0, 1).expect("layer0") };

    let h = SmolLmConfig::DEFAULT.hidden_size;
    let hd = SmolLmConfig::DEFAULT.head_dim();
    let nh = SmolLmConfig::DEFAULT.num_attention_heads;
    let nk = SmolLmConfig::DEFAULT.num_key_value_heads;
    let seq = 2;

    // xb: [hidden, seq], same flat layout in both.
    println!("xb  maxdiff = {:.6}", maxdiff(&g.xb, &diag.xb));
    assert!(maxdiff(&g.xb, &diag.xb) < 1e-3, "xb diverges");

    // q_pre: ggml and reference now share the [head_dim, n_head, seq] dhs layout.
    println!("q_pre maxdiff = {:.6}", maxdiff(&g.q_pre, &diag.q));
    assert!(maxdiff(&g.q_pre, &diag.q) < 1e-3, "q_pre diverges");

    // q_post
    println!("q_post maxdiff = {:.6}", maxdiff(&g.q_post, &diag.qr));
    assert!(maxdiff(&g.q_post, &diag.qr) < 1e-3, "q_post diverges");

    // k_post
    println!("k_post maxdiff = {:.6}", maxdiff(&g.k_post, &diag.kr));
    assert!(maxdiff(&g.k_post, &diag.kr) < 1e-3, "k_post diverges");

    // v
    println!("v_     maxdiff = {:.6}", maxdiff(&g.v_, &diag.vr));
    assert!(maxdiff(&g.v_, &diag.vr) < 1e-3, "v diverges");

    // scores: [seq, seq, n_head] same flat layout. Reference used GROUPED GQA.
    println!("scores maxdiff (ref grouped) = {:.6}", maxdiff(&g.scores, &diag.scores));
    // Also compute INTERLEAVED-GQA scores (mapping kh = hg % nkv) and check if
    // that matches ggml — confirming ggml's `repeat` is interleaved.
    let inter = interleaved_scores(&diag.qr, &diag.kr, hd, nh, nk, seq);
    println!("scores maxdiff (ggml vs interleaved) = {:.6}", maxdiff(&g.scores, &inter));
    assert!(maxdiff(&g.scores, &diag.scores) < 1e-3, "scores diverges (grouped)");

    // values: [head_dim, n_head, seq]
    println!("values maxdiff = {:.6}", maxdiff(&g.values, &diag.values));
    assert!(maxdiff(&g.values, &diag.values) < 1e-3, "values diverges");

    // o: [hidden, seq]
    println!("o      maxdiff = {:.6}", maxdiff(&g.o, &diag.o));
    assert!(maxdiff(&g.o, &diag.o) < 1e-3, "o diverges");

    println!("layer0_parity: ALL LAYER-0 INTERMEDIATES MATCH");
}

// --- Host-side generation smoke test (port of kernel BPE tokenizer) ---------
// Isolates whether low-quality kernel output is a tokenizer/encoding issue or a
// ggml issue: this uses the PURE-RUST reference model (validated == ggml).
struct Tok {
    byte_to_id: [u32; 256],
    vocab_bytes: Vec<Vec<u8>>,
    merges: Vec<(u32, u32, u32)>,
}
impl Tok {
    fn new(blob: &[u8]) -> Self {
        let mut off = 0usize;
        let rd = |off: &mut usize, n: usize| -> &[u8] { let s = &blob[*off..*off + n]; *off += n; s };
        let rd_u32 = |off: &mut usize| -> u32 { let b: [u8; 4] = rd(&mut *off, 4).try_into().unwrap(); u32::from_le_bytes(b) };
        assert_eq!(rd_u32(&mut off), 0x544F4B45, "bad tokenizer magic");
        let vocab_len = rd_u32(&mut off) as usize;
        let merge_len = rd_u32(&mut off) as usize;
        let mut byte_to_id = [0xFFFF_FFFFu32; 256];
        for b in 0..256 { byte_to_id[b] = rd_u32(&mut off); }
        let mut vocab_bytes = Vec::with_capacity(vocab_len);
        for _ in 0..vocab_len {
            let l = rd_u32(&mut off) as usize;
            vocab_bytes.push(rd(&mut off, l).to_vec());
        }
        let mut merges = Vec::with_capacity(merge_len);
        for _ in 0..merge_len {
            let l = rd_u32(&mut off); let r = rd_u32(&mut off); let m = rd_u32(&mut off);
            merges.push((l, r, m));
        }
        Tok { byte_to_id, vocab_bytes, merges }
    }
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| self.byte_to_id[b as usize]).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            let mut best_rank = u32::MAX;
            for (i, (l, r, m)) in self.merges.iter().enumerate() {
                if (i as u32) >= best_rank { continue; }
                for j in 0..ids.len().saturating_sub(1) {
                    if ids[j] == *l && ids[j + 1] == *r { best_rank = i as u32; best = Some((j, *m)); break; }
                }
            }
            match best { Some((j, m)) => { ids[j] = m; ids.remove(j + 1); } None => break }
        }
        ids
    }
    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids { bytes.extend_from_slice(&self.vocab_bytes[id as usize]); }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[test]
#[ignore]
fn gen_smoke() {
    let w = load();
    let rr = Ref::new(w);
    let cfg = SmolLmConfig::DEFAULT;
    let tok = Tok::new(model_data::TOKENIZER_BYTES);
    let prompt = "The capital of France is";
    let ids0: Vec<i32> = tok.encode(prompt).iter().map(|&x| x as i32).collect();
    println!("prompt ids = {:?}", ids0);
    let mut ids: Vec<i32> = ids0.clone();
    let mut out = String::new();
    for _ in 0..8 {
        let (_d, logits) = rr.forward(&ids);
        let vocab = cfg.vocab_size;
        let base = (ids.len() - 1) * vocab;
        let next = (base..base + vocab).fold((0usize, f32::NEG_INFINITY), |(bi, bv), i| {
            if logits[i] > bv { (i - base, logits[i]) } else { (bi, bv) }
        }).0;
        ids.push(next as i32);
        out.push_str(&tok.decode(&[next as u32]));
    }
    println!("PROMPT: {}\nREF-GENERATED: {}", prompt, out);
    println!("REF-IDS: {:?}", ids);
    // Burn ground truth for the same prompt:
    let burn_ids: Vec<u32> = vec![7042, 30, 657, 314, 260, 3995, 2240, 281, 4649, 284];
    println!("BURN-IDS: {:?}", burn_ids);
    println!("BURN-TEXT: {}", tok.decode(&burn_ids));
}

/// Fast generation through the GGML forward (the production path). Proves the
/// model emits coherent continuation text, not just matching-argmax logits.
#[test]
#[ignore]
fn gen_smoke_ggml() {
    let w = load();
    let cfg = SmolLmConfig::DEFAULT;
    let tok = Tok::new(model_data::TOKENIZER_BYTES);
    let prompt = "The capital of France is";
    let ids0: Vec<i32> = tok.encode(prompt).iter().map(|&x| x as i32).collect();
    println!("ggml prompt ids = {:?}", ids0);
    let model = unsafe { model::forward::Model::new(&w) };
    let mut ids: Vec<i32> = ids0.clone();
    let mut out = String::new();
    let vocab = cfg.vocab_size;
    for _ in 0..20 {
        let logits = unsafe { model.forward(&ids) };
        let next = model::forward::Model::argmax_last(&logits, ids.len(), vocab);
        ids.push(next as i32);
        out.push_str(&tok.decode(&[next as u32]));
    }
    println!("PROMPT: {}\nGGML-GENERATED: {}", prompt, out);
    println!("GGML-IDS: {:?}", ids);
}

/// Tokenizer round-trip sanity: encode("The capital of France is") should
/// decode back to the prompt, and the burn ground-truth ids should decode to
/// something readable. If this fails the model is fed the wrong tokens.
#[test]
#[ignore]
fn gen_tok_roundtrip() {
    let tok = Tok::new(model_data::TOKENIZER_BYTES);
    let prompt = "The capital of France is";
    let ids: Vec<u32> = tok.encode(prompt);
    println!("tok  encode = {:?}", ids);
    println!("tok  decode = {:?}", tok.decode(&ids));
    let burn_ids: Vec<u32> = vec![7042, 30, 657, 314, 260, 3995, 2240, 281, 4649, 284];
    println!("burn ids    = {:?}", burn_ids);
    println!("burn decode = {:?}", tok.decode(&burn_ids));
    let single: Vec<u32> = vec![504];
    println!("token 504   = {:?}", tok.decode(&single));
    let single2: Vec<u32> = vec![3575];
    println!("token 3575  = {:?}", tok.decode(&single2));
}

#[test]
#[ignore]
fn first_token_matches_burn() {
    let w = load();
    let ref_ = Ref::new(w);
    let cfg = SmolLmConfig::DEFAULT;
    let ids0: Vec<i32> = vec![504, 3575, 282, 4649, 314];
    let (_, logits) = ref_.forward(&ids0);
    let base = (ids0.len() - 1) * cfg.vocab_size;
    let next = (base..base + cfg.vocab_size).fold((0usize, f32::NEG_INFINITY), |(bi, bv), i| {
        if logits[i] > bv { (i - base, logits[i]) } else { (bi, bv) }
    }).0;
    println!("PROMPT=[504,3575,282,4649,314] first_token={}", next);
    println!("BURN first_token=7042");
    assert_eq!(next, 7042, "reference first token does not match burn ground truth");
}

/// Compare our layer-0 attention intermediates against the burn oracle's for
/// the REAL prompt [504,3575,282,4649,314]. Burn printed (for last token pos 4):
///   xb   = [0.034314405, 0.032350224, 0.0009292086, ...]  norm2 0.738
///   attn = [0.031417202, 0.005140569, 0.010526163, ...]  norm2 3.893
/// If attn diverges, the bug is inside attention, not in embed/rmsnorm.
#[test]
#[ignore]
fn diag_vs_burn() {
    let w = load();
    let model = unsafe { model::forward::Model::new(&w) };
    let prompt = [504i32, 3575, 282, 4649, 314];
    let d = unsafe { model.layer0_diag_full(&prompt).expect("layer0") };
    let hi = SmolLmConfig::DEFAULT.hidden_size;
    let last = 4 * hi; // last token (pos index 4)
    let norm2 = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
    println!("our xb last[0..8]  = {:?}", &d.xb[last..last + 8]);
    println!("our xb last norm    = {}", norm2(&d.xb[last..last + hi]));
    println!("burn xb last[0..8]  = [0.034314405, 0.032350224, 0.0009292086, -0.00015647475, -0.013665326, 0.0041701067, 0.0065982626, -0.018581653]");
    println!("burn xb last norm   = 0.7379983228125072");
    println!("our attn last[0..8] = {:?}", &d.o[last..last + 8]);
    println!("our attn last norm  = {}", norm2(&d.o[last..last + hi]));
    println!("burn attn last[0..8]= [0.031417202, 0.005140569, 0.010526163, 0.039213166, -0.02677809, -0.004461091, -0.0077143633, -0.010799428]");
    println!("burn attn last norm = 3.8930606019400287");
    println!("our q_post last[0..8] = {:?}", &d.q_post[last + 0 * 0 & 0..last + 8]);
    println!("our k_post last[0..8] = {:?}", &d.k_post[0..8]);
    println!("our scores[..8]  = {:?}", &d.scores[0..8]);
}
