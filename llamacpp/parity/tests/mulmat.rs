//! Tiny, model-free test to settle the ggml mul_mat orientation
//! (W@x vs W^T@x vs transposed-layout) and the flat index convention of the
//! output tensor. No heavy model is loaded here.
use ggml_sys::*;

fn flat(t: *mut ggml_tensor) -> Vec<f32> {
    unsafe {
        let p = ggml_get_data_f32(t);
        let n = (ggml_nbytes(t) / 4) as usize;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

#[test]
fn mul_mat_orientation() {
    // 2x3 weight: W[o][i]. Use distinguishable values.
    // W = [ [1,2,3],
    //       [4,5,6] ]      (o=0 row = 1,2,3 ; o=1 row = 4,5,6)
    // x (in=3, seq=1) = [[10],[20],[30]]
    // W@x  = [1*10+2*20+3*30, 4*10+5*20+6*30] = [140, 320]
    let w: [f32; 6] = [1., 2., 3., 4., 5., 6.];
    let x: [f32; 3] = [10., 20., 30.];

    let ctx = unsafe {
        ggml_init(ggml_init_params {
            mem_size: 1 << 20,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        })
    };

    // Store W as [ne0=in, ne1=out] flat (what weight_2d does).
    let wt = unsafe { ggml_new_tensor_2d(ctx, ggml_type::F32, 3, 2) };
    unsafe {
        let d = ggml_get_data_f32(wt);
        for i in 0..6 {
            *d.add(i) = w[i];
        }
    }
    let xt = unsafe { ggml_new_tensor_2d(ctx, ggml_type::F32, 3, 1) };
    unsafe {
        let d = ggml_get_data_f32(xt);
        for i in 0..3 {
            *d.add(i) = x[i];
        }
    }
    let c = unsafe { ggml_mul_mat(ctx, wt, xt) };
    let gf = unsafe { ggml_new_graph(ctx) };
    unsafe { ggml_build_forward_expand(gf, c) };
    unsafe { ggml_graph_compute_with_ctx(ctx, gf, 1) };
    let out = flat(c);
    // Print raw flat and the W@x / W^T@x expected values.
    println!("mul_mat out flat = {:?}", out);
    println!("expected W@x (o-major)  = [140, 320]");
    println!("expected x@W / row-major matmul = [140, 320] too (same square-ish)");
    // So c should read [o + s*ne0] with ne0 = out = 2 -> c[o,0] = raw[o].
    assert!(
        (out[0] - 140.0).abs() < 1e-3 && (out[1] - 320.0).abs() < 1e-3,
        "mul_mat did not yield W@x under [ne0=in,ne1=out] storage: {:?}",
        out
    );
}

/// Pin down ggml_repeat's head (ne2) tiling. k = [ne0=hd, ne1=seq, ne2=nk],
/// repeated to match q = [ne0=hd, ne1=seq, ne2=nh]. Use distinguishable values
/// so we can see which source head each output head comes from.
#[test]
fn repeat_head_ordering() {
    let hd = 2usize;
    let seq = 2usize;
    let nk = 2usize; // kv heads
    let nh = 4usize; // query heads (2x repeat)

    let ctx = unsafe {
        ggml_init(ggml_init_params {
            mem_size: 1 << 20,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        })
    };

    // k stored as [ne0=hd, ne1=seq, ne2=nk]; fill per-head constant: base 100 + h.
    let k = unsafe { ggml_new_tensor_3d(ctx, ggml_type::F32, hd as i64, seq as i64, nk as i64) };
    unsafe {
        let d = ggml_get_data_f32(k);
        for h in 0..nk {
            for s in 0..seq {
                for dd in 0..hd {
                    let idx = dd + s * hd + h * hd * seq;
                    *d.add(idx) = (100 + h) as f32 + (dd * 10 + s) as f32;
                }
            }
        }
    }
    // q: any [ne0=hd, ne1=seq, ne2=nh] to define the target shape.
    let q = unsafe { ggml_new_tensor_3d(ctx, ggml_type::F32, hd as i64, seq as i64, nh as i64) };
    let r = unsafe { ggml_repeat(ctx, k, q) };
    let gf = unsafe { ggml_new_graph(ctx) };
    unsafe { ggml_build_forward_expand(gf, r) };
    unsafe { ggml_graph_compute_with_ctx(ctx, gf, 1) };
    let out = flat(r);
    // Expect length = hd*seq*nh = 16. Print per output head (first value of the
    // head's (s=0,d=0) slot), which reveals the source head by the base 100+h.
    println!("repeat (hd=2,seq=2,nk=2->nh=4): flat = {:?}", out);
    for h in 0..nh {
        let v = out[0 + 0 * hd + h * hd * seq]; // (d=0,s=0,h)
        println!("  out head {} source value {:.0} -> source head {:.0}", h, v, v - 100.0);
    }
}

