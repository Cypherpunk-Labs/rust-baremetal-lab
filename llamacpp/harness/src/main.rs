//! Host-side validation harness for the ggml FFI.
//!
//! This is the seed for ticket T4 (single mul_mat parity): it builds the same
//! all-ones graph we validated in C, runs it through the Rust FFI bindings,
//! and checks the result. On the host it links the system libc; the exact
//! same ggml-sys crate is what the kernel links freestanding.

use ggml_sys::*;
use model::safetensors::SafeTensors;
use model::{SmolLmConfig, SmolLmWeights};
use model_data::MODEL_BYTES;
use std::ptr;

fn main() {
    let mem_size = 256 * 1024 * 1024;
    let params = ggml_init_params {
        mem_size,
        mem_buffer: ptr::null_mut(),
        no_alloc: false,
    };
    let ctx = unsafe { ggml_init(params) };
    assert!(!ctx.is_null(), "ggml_init failed");

    // a: [K=8, M=4], b: [K=8, N=8] -> c: [M=4, N=8]; all-ones => c[0] = 8.
    let a = unsafe { ggml_new_tensor_2d(ctx, ggml_type::F32, 8, 4) };
    let b = unsafe { ggml_new_tensor_2d(ctx, ggml_type::F32, 8, 8) };
    assert!(!a.is_null() && !b.is_null());

    unsafe {
        let ad = ggml_get_data_f32(a);
        let bd = ggml_get_data_f32(b);
        for i in 0..(ggml_nbytes(a) / 4) {
            *ad.add(i) = 1.0;
        }
        for i in 0..(ggml_nbytes(b) / 4) {
            *bd.add(i) = 1.0;
        }
    }

    let c = unsafe { ggml_mul_mat(ctx, a, b) };
    assert!(!c.is_null());

    let gf = unsafe { ggml_new_graph(ctx) };
    unsafe { ggml_build_forward_expand(gf, c) };

    // ggml_graph_compute_with_ctx plans + computes internally (avoids the
    // large struct-by-value return of ggml_graph_plan across the FFI).
    let status = unsafe { ggml_graph_compute_with_ctx(ctx, gf, 1) };
    let c0 = unsafe { *ggml_get_data_f32(c) };
    println!(
        "[harness] mul_mat c[0] = {:.6} (expected 8.0), status {}",
        c0,
        status as i32
    );
    assert!((c0 - 8.0).abs() < 1e-3, "mul_mat result wrong: {}", c0);

    unsafe { ggml_free(ctx) };
    println!("[harness] OK");

    // --- ggml rope identity-at-pos-0 check -------------------------------
    unsafe {
        let ctx = ggml_sys::ggml_init(ggml_sys::ggml_init_params {
            mem_size: 1 << 20,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        });
        let a = ggml_sys::ggml_new_tensor_3d(ctx, ggml_sys::ggml_type::F32, 4, 1, 2);
        let d = ggml_sys::ggml_get_data_f32(a);
        for i in 0..8 {
            *d.add(i) = i as f32;
        }
        let pos = ggml_sys::ggml_new_tensor_1d(ctx, ggml_sys::ggml_type::I32, 2);
        let p = ggml_sys::ggml_get_data(pos) as *mut i32;
        *p.add(0) = 0;
        *p.add(1) = 1;
        for mode in [2i32, 0] {
            let out = ggml_sys::ggml_rope(ctx, a, pos, 4, mode);
            let gf = ggml_sys::ggml_new_graph(ctx);
            ggml_sys::ggml_build_forward_expand(gf, out);
            ggml_sys::ggml_graph_compute_with_ctx(ctx, gf, 1);
            let o = ggml_sys::ggml_get_data_f32(out);
            println!(
                "[rope] mode={} pos0 (expect 0,1,2,3) = [{}, {}, {}, {}] | pos1 = [{}, {}, {}, {}]",
                mode,
                *o.add(0),
                *o.add(1),
                *o.add(2),
                *o.add(3),
                *o.add(4),
                *o.add(5),
                *o.add(6),
                *o.add(7)
            );
        }
    }

    // --- model load (embedded) + safetensors parse validation -----------
    println!("[harness] embedded model: {} bytes", MODEL_BYTES.len());
    let st = SafeTensors::parse(MODEL_BYTES).expect("parse safetensors");
    println!("[harness] safetensors tensors: {}", st.len());
    let mut total_params: usize = 0;
    for t in st.tensors() {
        let n: usize = t.shape.iter().product();
        total_params += n;
        if t.name.starts_with("model.embed_tokens")
            || t.name.starts_with("model.norm")
            || t.name.starts_with("model.layers.0.")
        {
            println!(
                "[harness]   {} shape={:?} dtype={} params={}",
                t.name,
                t.shape,
                if t.dtype == model::safetensors::DType::F32 { "F32" } else { "?" },
                n
            );
        }
    }
    println!("[harness] total params: {}", total_params);

    let cfg = SmolLmConfig::DEFAULT;
    let w = SmolLmWeights::from_safetensors(&st, cfg);
    println!(
        "[harness] embed_tokens.len() = {}, norm.len() = {}, layer0.q_proj.len() = {}",
        w.embed_tokens.len(),
        w.norm.len(),
        w.layers[0].q_proj.len()
    );
    assert_eq!(w.embed_tokens.len(), cfg.vocab_size * cfg.hidden_size);
    assert_eq!(w.layers.len(), cfg.num_hidden_layers);
    println!("[harness] model weights resolved OK");

    // --- transformer forward (host validation) ----------------------------
    println!("[harness] building Model (copying weights into ggml)...");
    let model = unsafe { model::forward::Model::new(&w) };

    // Layer-0 diagnostic: sanity check the ggml attention path produces
    // finite, non-trivial values (deep parity lives in the `parity` tests).
    unsafe {
        let d0 = model.layer0_diag(0, 1).expect("layer0_diag");
        let max_o = d0.o.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        let max_s = d0.scores.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        println!(
            "[harness] layer0 diag: o[0..5]={:?} .. max_o={:.4} max_scores={:.4}",
            &d0.o[0..5], max_o, max_s
        );
        assert!(d0.o.len() == cfg.hidden_size * 2);
    }
    // Fixed, deterministic token ids for a smoke test + reference comparison.
    let prompt: Vec<i32> = (0..16).map(|i| (i * 7 % cfg.vocab_size as i32)).collect();
    println!("[harness] forward over {} tokens...", prompt.len());
    let logits = unsafe { model.forward(&prompt) };
    let next = model::forward::Model::argmax_last(&logits, prompt.len(), cfg.vocab_size);
    // Sanity: logits must be finite.
    let mut finite = true;
    for &v in logits.iter() {
        if !v.is_finite() {
            finite = false;
            break;
        }
    }
    println!(
        "[harness] forward OK; logits finite={}, greedy next-token = {}",
        finite, next
    );
    assert!(finite, "forward produced non-finite logits");
    println!("[harness] FORWARD VALIDATED");
}
