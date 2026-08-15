#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use burn_core::{
    module::Module,
    record::{NoStdInferenceRecorder, Recorder},
    tensor::{Int, Shape, Tensor, TensorData},
};
use core::alloc::{GlobalAlloc, Layout};
use core::arch::{asm, global_asm};
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::panic::PanicInfo;
use kernel::model::{SmolLmConfig, SmolLmModel, SmolLmModelRecord};
use linked_list_allocator::Heap;

type Backend = burn_flex::Flex;

const HEAP_START: usize = 0x7000_0000;
const HEAP_SIZE: usize = 0x8000_0000;

struct KernelHeap(UnsafeCell<Heap>);

unsafe impl Sync for KernelHeap {}

const fn kernel_heap() -> KernelHeap {
    KernelHeap(UnsafeCell::new(Heap::empty()))
}

unsafe impl GlobalAlloc for KernelHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            (*self.0.get())
                .allocate_first_fit(layout)
                .map(|p| p.as_ptr())
                .unwrap_or(core::ptr::null_mut())
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            (*self.0.get()).deallocate(core::ptr::NonNull::new_unchecked(ptr), layout);
        }
    }
}

#[global_allocator]
static ALLOCATOR: KernelHeap = kernel_heap();

static MODEL_BYTES: &[u8] = include_bytes!("smollm-135m.bin");

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
    // Point VBAR_EL1 at our exception vector table so any exception
    // lands in a handler that dumps ESR/ELR/FAR instead of spinning at 0x200.
    adrp x0, exception_vector_base
    add  x0, x0, :lo12:exception_vector_base
    msr  vbar_el1, x0
    isb
    bl   kernel_main
1:  b    1b
    "#
);

// Exception vector table (AArch64, EL1). VBAR_EL1 must be 2KiB aligned.
// Each entry is 0x80 bytes apart; all route to exception_entry which
// captures ESR_EL1 / ELR_EL1 / FAR_EL1 and prints them to the UART.
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
    // x0..x2 = ESR_EL1, ELR_EL1, FAR_EL1 (caller-saved, fine to clobber)
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

fn cntpct() -> u64 {
    let mut v: u64 = 0;
    unsafe {
        asm!("mrs {0}, cntpct_el0", out(reg) v);
    }
    v
}

fn cntfrq() -> u64 {
    let mut v: u64 = 0;
    unsafe {
        asm!("mrs {0}, cntfrq_el0", out(reg) v);
    }
    v
}

struct Pl011;

impl Pl011 {
    fn init(&self) {
        unsafe {
            UART_CR.write_volatile(0x301);
        }
    }

    fn putc(&self, c: u8) {
        unsafe {
            while UART_FR.read_volatile() & FR_TXFF != 0 {}
            UART_DR.write_volatile(c as u32);
        }
    }
}

impl Write for Pl011 {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for b in s.bytes() {
            match b {
                b'\n' => {
                    self.putc(b'\r');
                    self.putc(b'\n');
                }
                _ => self.putc(b),
            }
        }
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn kernel_main() -> ! {
    let mut serial = Pl011;
    serial.init();

    unsafe {
        kernel::mmu::init(&mut serial);
        (*ALLOCATOR.0.get()).init(HEAP_START as *mut u8, HEAP_SIZE);
    }

    writeln!(
        serial,
        "[Kernel] {} GiB heap initialized",
        HEAP_SIZE / (1024 * 1024 * 1024)
    )
    .ok();

    // --- MINIMAL ATOMICS PROBE: does a bare ldaxrb/stxrb work under HVF? ---
    let probe: &mut u8 = unsafe { &mut *(HEAP_START as *mut u8) };
    *probe = 0;
    let mut prev: u32;
    let mut status: u32;
    unsafe {
        core::arch::asm!(
            "ldaxrb {r:w}, [{addr}]",
            "stlxrb {s:w}, {val:w}, [{addr}]",
            r = out(reg) prev,
            s = out(reg) status,
            addr = in(reg) probe as *mut u8 as usize,
            val = in(reg) 1u32,
        );
    }
    writeln!(serial, "[Kernel] atomics probe: prev={} status={}", prev, status).ok();
    if status != 0 {
        writeln!(serial, "[Kernel] stlxrb FAILED (status != 0) -> atomics broken").ok();
    } else {
        writeln!(serial, "[Kernel] stlxrb succeeded -> atomics OK").ok();
    }
    // --------------------------------------------------------------------

    writeln!(serial, "[Kernel] Loading SmolLM-135M weights from embedded bytes...").ok();

    let device = Default::default();
    let config = SmolLmConfig::default();

    let recorder = NoStdInferenceRecorder::new();
    let record = match recorder.load::<SmolLmModelRecord<Backend>>(MODEL_BYTES, &device) {
        Ok(record) => {
            writeln!(serial, "[Kernel] Weights deserialized successfully").ok();
            record
        }
        Err(_) => {
            writeln!(serial, "[Kernel ERROR] Failed to parse model binary").ok();
            loop {}
        }
    };

    let model = SmolLmModel::<Backend>::new(&config, &device).load_record(record);
    writeln!(serial, "[Kernel] Model loaded onto burn-flex backend").ok();

    let mut input_ids: Vec<i64> = vec![15496, 11];

    writeln!(serial, "[Kernel] Starting autoregressive generation loop (greedy)...").ok();
    let freq = cntfrq();
    let mut total_ms: u64 = 0;

    for step in 0..10 {
        let seq_len = input_ids.len();
        let data = TensorData::new(input_ids.clone(), Shape::new([1, seq_len]));
        let input_tensor = Tensor::<Backend, 2, Int>::from_data(data, &device);

        let t0 = cntpct();
        let logits = model.forward(input_tensor);
        let last_logits = logits.slice([0..1, seq_len - 1..seq_len]);
        let next_token = last_logits.argmax(2).into_scalar() as i64;
        let t1 = cntpct();
        input_ids.push(next_token);

        let step_ms = (t1 - t0) * 1000 / freq;
        total_ms += step_ms;
        writeln!(serial, "[step {}] token={} ({} ms)", step, next_token, step_ms).ok();
    }

    let tps = (10.0 * 1000.0) / (total_ms as f64);
    writeln!(
        serial,
        "[Kernel] avg {:.2} ms/token, {:.3} tokens/sec (cntfrq={} Hz)",
        total_ms as f64 / 10.0,
        tps,
        freq
    )
    .ok();
    writeln!(serial, "[Kernel] Inference test completed successfully").ok();
    loop {}
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let mut serial = Pl011;
    serial.init();
    writeln!(serial, "[Kernel PANIC] {}", info).ok();
    loop {}
}

/// Called from the exception vector entry. Dumps the exception syndrome,
/// faulting PC and fault address to the UART so we can see exactly which
/// exception HVF is taking. Never returns (spins).
#[no_mangle]
fn dump_exception(esr: u64, elr: u64, far: u64) -> ! {
    let mut serial = Pl011;
    serial.init();
    writeln!(serial, "[Kernel EXCEPTION]").ok();
    writeln!(serial, "  ESR_EL1 = {:#018x}", esr).ok();
    writeln!(serial, "  ELR_EL1 = {:#018x}", elr).ok();
    writeln!(serial, "  FAR_EL1 = {:#018x}", far).ok();
    let ec = esr >> 26 & 0x3f;
    writeln!(serial, "  EC      = {:#04x} ({})", ec, ec_name(ec)).ok();
    loop {}
}

fn ec_name(ec: u64) -> &'static str {
    match ec {
        0x00 => "Unknown reason",
        0x01 => "Trapped WFI/WFE",
        0x02 => "Trapped MCR/MRC",
        0x03 => "Trapped MCRR/MRRC",
        0x04 => "Trapped MCRC",
        0x05 => "Trapped LDC/STC",
        0x06 => "Trapped LDNP/STNP",
        0x07 => "Trapped FP/SIMD access",
        0x08 => "Trapped PState change",
        0x0c => "Trapped SVC (AArch64)",
        0x15 => "SVE access trap",
        0x16 => "Trapped ERET/ERETAA/ERETAB",
        0x18 => "PAC exception",
        0x20 => "Instruction abort (same EL)",
        0x21 => "Instruction abort (lower EL)",
        0x22 => "PC alignment fault",
        0x24 => "Data abort (same EL)",
        0x25 => "Data abort (lower EL)",
        0x26 => "SP alignment fault",
        0x2c => "Trapped FP exception",
        0x34 => "Data cache maintenance (same EL)",
        _ => "Other",
    }
}