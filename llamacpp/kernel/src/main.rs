#![no_std]
#![no_main]

extern crate alloc;

use core::alloc::{GlobalAlloc, Layout};
use core::arch::global_asm;
use core::ffi::c_char;
use core::fmt::Write;
use core::panic::PanicInfo;

use os_api::install_uart_putchar;

// The heap lives at 0x7000_0000 (2 GiB) and is carved by Newlib's `malloc`
// via the `_sbrk` syscall in `os-api`. Rust allocations and ggml allocations
// therefore share one bump region.
extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

struct NewlibAlloc;
unsafe impl Sync for NewlibAlloc {}
unsafe impl GlobalAlloc for NewlibAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        malloc(layout.size())
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        free(ptr)
    }
}
#[global_allocator]
static ALLOC: NewlibAlloc = NewlibAlloc;

global_asm!(
    r#"
    .section .text._start
    .globl _start
    _start:
        mov  x0, #3
        lsl  x0, x0, #20
        msr  cpacr_el1, x0
        isb
        adrp x0, __stack_top
        add  x0, x0, :lo12:__stack_top
        mov  sp, x0
        adrp x0, exception_vector_base
        add  x0, x0, :lo12:exception_vector_base
        msr  vbar_el1, x0
        isb
        bl   kernel_main
    1:  b    1b
    "#
);

global_asm!(
    r#"
    .section .text.vectors, "ax"
    .balign 0x800
    .globl exception_vector_base
    exception_vector_base:
        .rept 16
        b   exception_entry
        .balign 0x80
        .endr
    "#
);

global_asm!(
    r#"
    .section .text.exception_entry, "ax"
    .globl exception_entry
    exception_entry:
        mrs x0, esr_el1
        mrs x1, elr_el1
        mrs x2, far_el1
        bl  dump_exception
    1:  b   1b
    "#
);

const UART_BASE: usize = 0x0900_0000;
const UART_DR: *mut u32 = UART_BASE as *mut u32;
const UART_FR: *const u32 = (UART_BASE + 0x18) as *const u32;
const UART_CR: *mut u32 = (UART_BASE + 0x30) as *mut u32;
const FR_TXFF: u32 = 1 << 5;

unsafe extern "C" fn uart_putc_c(c: c_char) {
    while UART_FR.read_volatile() & FR_TXFF != 0 {}
    UART_DR.write_volatile(c as u32);
}

struct Serial;
impl Write for Serial {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            match b {
                b'\n' => {
                    unsafe {
                        uart_putc_c(b'\r' as c_char);
                        uart_putc_c(b'\n' as c_char);
                    }
                }
                _ => unsafe { uart_putc_c(b as c_char) },
            }
        }
        Ok(())
    }
}

const UART_FR_RXFE: u32 = 1 << 4; // receive FIFO empty

/// Blocking read of one byte from the PL011 UART.
unsafe extern "C" fn uart_getc_c() -> u8 {
    while UART_FR.read_volatile() & UART_FR_RXFE != 0 {}
    UART_DR.read_volatile() as u8
}

/// Read a line from the UART (until LF/CR or Ctrl-D) into a String, echoing.
fn read_line() -> alloc::string::String {
    let mut s = alloc::string::String::new();
    loop {
        let c = unsafe { uart_getc_c() };
        match c {
            b'\r' | b'\n' => {
                let _ = writeln!(Serial, "");
                break;
            }
            b'\x04' => break, // Ctrl-D
            0x7f | 0x08 => {
                // backspace
                if !s.is_empty() {
                    s.pop();
                    unsafe {
                        uart_putc_c(0x08u8 as c_char);
                        uart_putc_c(b' ' as c_char);
                        uart_putc_c(0x08u8 as c_char);
                    }
                }
            }
            _ => {
                s.push(c as char);
                unsafe { uart_putc_c(c as c_char) };
            }
        }
    }
    s
}

/// Minimal f32 -> decimal formatter (no libm). Good enough for diag output.
fn print_f32(v: f32) -> alloc::string::String {
    if v < 0.0 {
        return alloc::format!("-{}", print_f32(-v));
    }
    let i = v as u64;
    let frac = ((v - i as f32) * 1000.0) as u64;
    alloc::format!("{}.{:03}", i, frac)
}

/// T1-kernel seed: run one ggml mul_mat through the FFI on the embedded heap.
unsafe fn ggml_hello() {
    use ggml_sys::*;
    let params = ggml_init_params {
        mem_size: 64 * 1024 * 1024,
        mem_buffer: core::ptr::null_mut(),
        no_alloc: false,
    };
    let ctx = ggml_init(params);
    if ctx.is_null() {
        writeln!(Serial, "[ggml] init FAILED").ok();
        return;
    }
    let a = ggml_new_tensor_2d(ctx, ggml_type::F32, 8, 4);
    let b = ggml_new_tensor_2d(ctx, ggml_type::F32, 8, 8);
    let ad = ggml_get_data_f32(a);
    let bd = ggml_get_data_f32(b);
    for i in 0..(ggml_nbytes(a) / 4) {
        *ad.add(i) = 1.0;
    }
    for i in 0..(ggml_nbytes(b) / 4) {
        *bd.add(i) = 1.0;
    }
    let c = ggml_mul_mat(ctx, a, b);
    let gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    let st = ggml_graph_compute_with_ctx(ctx, gf, 1);
    let c0 = *ggml_get_data_f32(c);
    writeln!(
        Serial,
        "[ggml] mul_mat c0 = {} (status {})",
        print_f32(c0),
        st as i32
    )
    .ok();
    ggml_free(ctx);
}

#[no_mangle]
pub extern "C" fn kernel_main() -> ! {
    unsafe {
        UART_CR.write_volatile(0x301);
    }
    install_uart_putchar(uart_putc_c);

    unsafe {
        kernel::mmu::init();
    }

    writeln!(Serial, "[Kernel] MMU on; heap via Newlib malloc").ok();

    // T1: first ggml call through the freestanding FFI.
    unsafe { ggml_hello() };

    // --- SmolLM-135M on-device chat loop -----------------------------------
    let tok = kernel::tokenizer::Tokenizer::new(model_data::TOKENIZER_BYTES);
    let weights = match model::safetensors::SafeTensors::parse(model_data::MODEL_BYTES) {
        Ok(st) => st,
        Err(e) => {
            writeln!(Serial, "[Kernel] model parse FAILED: {}", e).ok();
            loop {}
        }
    };
    let w = model::SmolLmWeights::from_safetensors(&weights, model::SmolLmConfig::DEFAULT);
    let model = unsafe { model::forward::Model::new(&w) };
    let vocab = model::SmolLmConfig::DEFAULT.vocab_size;

    writeln!(Serial, "[Kernel] SmolLM-135M ready. Type a prompt, Ctrl-D to exit.").ok();

    const MAX_NEW: usize = 48;
    let mut history: alloc::vec::Vec<i32> = alloc::vec::Vec::new();
    loop {
        write!(Serial, "You> ").ok();
        let line = read_line();
        if line.is_empty() {
            continue;
        }
        // Encode the new user turn and append to the running history.
        let ids: alloc::vec::Vec<i32> =
            tok.encode(&line).iter().map(|&x| x as i32).collect();
        history.extend_from_slice(&ids);

        // Greedy-decode up to MAX_NEW tokens, taking argmax of the last position.
        write!(Serial, "Bot> ").ok();
        let mut generated = 0usize;
        loop {
            let logits = unsafe { model.forward(&history) };
            let next = model::forward::Model::argmax_last(&logits, history.len(), vocab);
            history.push(next as i32);
            let piece = tok.decode(&[next as u32]);
            write!(Serial, "{}", piece).ok();
            generated += 1;
            if generated >= MAX_NEW || piece.contains('\n') {
                break;
            }
        }
        writeln!(Serial, "").ok();
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let _ = writeln!(Serial, "[Kernel PANIC] {}", info);
    loop {}
}

#[no_mangle]
fn dump_exception(esr: u64, elr: u64, far: u64) -> ! {
    let _ = writeln!(
        Serial,
        "[Kernel EXCEPTION] ESR={:#x} ELR={:#x} FAR={:#x}",
        esr, elr, far
    );
    loop {}
}
