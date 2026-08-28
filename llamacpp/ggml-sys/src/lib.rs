//! Raw FFI bindings to the freestanding ggml CPU backend.
//!
//! These are hand-written (no bindgen, so the crate builds for
//! `aarch64-unknown-none` without a host libclang). Only the symbols the
//! kernel/harness actually call are declared; extend as tickets T3/T4 need
//! more of the ggml/gguf surface.

#![no_std]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use core::ffi::{c_int, c_void};

// Opaque ggml handles.
pub enum ggml_context {}
pub enum ggml_tensor {}
pub enum ggml_cgraph {}
pub enum ggml_backend {}
pub enum ggml_backend_buffer {}
pub enum ggml_backend_buffer_type {}
pub enum ggml_threadpool {}
pub enum gguf_context {}

pub type ggml_backend_t = *mut ggml_backend;
pub type ggml_backend_buffer_t = *mut ggml_backend_buffer;
pub type ggml_backend_buffer_type_t = *mut ggml_backend_buffer_type;
pub type ggml_abort_callback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum ggml_type {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
    COUNT = 43,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub enum ggml_status {
    GGML_STATUS_SUCCESS = 0,
    GGML_STATUS_FAILED = 1,
    GGML_STATUS_ALLOC_FAILED = 2,
    GGML_STATUS_INPUT_CHANGED = 4,
    GGML_STATUS_ABORTED = 5,
    GGML_STATUS_ABORTING = 6,
}

#[repr(C)]
pub struct ggml_init_params {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

// Must match `struct ggml_cplan` in ggml-cpu.h exactly (field order/types).
#[repr(C)]
pub struct ggml_cplan {
    pub work_size: usize,
    pub work_data: *mut u8,
    pub n_threads: c_int,
    pub threadpool: *mut ggml_threadpool,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut c_void,
    pub use_ref: bool,
}

#[repr(C)]
pub struct gguf_init_params {
    pub no_alloc: bool,
    pub ctx: *mut *mut ggml_context,
}

extern "C" {
    pub fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
    pub fn ggml_free(ctx: *mut ggml_context);
    pub fn ggml_get_data(tensor: *const ggml_tensor) -> *mut c_void;
    pub fn ggml_get_data_f32(tensor: *const ggml_tensor) -> *mut f32;
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;
    pub fn ggml_nrows(tensor: *const ggml_tensor) -> usize;

    pub fn ggml_new_tensor_1d(ctx: *mut ggml_context, t: ggml_type, ne0: i64) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        t: ggml_type,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_new_tensor_3d(
        ctx: *mut ggml_context,
        t: ggml_type,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;

    pub fn ggml_mul_mat(ctx: *mut ggml_context, a: *mut ggml_tensor, b: *mut ggml_tensor)
        -> *mut ggml_tensor;
    pub fn ggml_mul_mat_set_prec(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        prec: c_int,
    );

    pub fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;
    pub fn ggml_build_forward_expand(gf: *mut ggml_cgraph, tensor: *mut ggml_tensor);
    // NOTE: do NOT bind `ggml_graph_plan`/`ggml_graph_compute` directly - they
    // return/pass `struct ggml_cplan` by value (a 56-byte struct), whose sret
    // ABI is unsafe to cross from Rust. Use `ggml_graph_compute_with_ctx`,
    // which plans and computes internally and returns a plain `ggml_status`.
    pub fn ggml_graph_compute_with_ctx(
        ctx: *mut ggml_context,
        gf: *mut ggml_cgraph,
        n_threads: c_int,
    ) -> ggml_status;

    // T3: load GGUF from an in-RAM buffer (no mmap).
    pub fn gguf_init_from_buffer(
        data: *const c_void,
        size: usize,
        params: gguf_init_params,
    ) -> *mut gguf_context;
    pub fn gguf_free(ctx: *mut gguf_context);

    // T1/T3: CPU backend + zero-copy buffer wrapping externally-held RAM.
    pub fn ggml_backend_cpu_init() -> ggml_backend_t;
    pub fn ggml_backend_cpu_buffer_from_ptr(ptr: *mut c_void, size: usize)
        -> ggml_backend_buffer_t;
    pub fn ggml_backend_cpu_buffer_type() -> ggml_backend_buffer_type_t;

    // T5: transformer-forward ops (hand-written ggml graph, per spike plan).
    pub fn ggml_get_rows(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_rms_norm(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        weight: *mut ggml_tensor,
        eps: f32,
    ) -> *mut ggml_tensor;
    pub fn ggml_reshape_3d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_reshape_2d(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_transpose(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_rope(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
        n_dims: c_int,
        mode: c_int,
    ) -> *mut ggml_tensor;
    pub fn ggml_permute(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        axis0: c_int,
        axis1: c_int,
        axis2: c_int,
        axis3: c_int,
    ) -> *mut ggml_tensor;
    pub fn ggml_scale(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        s: f32,
    ) -> *mut ggml_tensor;
    pub fn ggml_soft_max(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_silu(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_mul(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_add(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_cont(ctx: *mut ggml_context, a: *mut ggml_tensor) -> *mut ggml_tensor;
    pub fn ggml_repeat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
}
