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
use core::arch::global_asm;
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
    bl   kernel_main
1:  b    1b
    "#
);

const UART_BASE: usize = 0x0900_0000;
const UART_DR: *mut u32 = UART_BASE as *mut u32;
const UART_FR: *const u32 = (UART_BASE + 0x18) as *const u32;
const UART_CR: *mut u32 = (UART_BASE + 0x30) as *mut u32;
const FR_TXFF: u32 = 1 << 5;

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
    unsafe {
        (*ALLOCATOR.0.get()).init(HEAP_START as *mut u8, HEAP_SIZE);
    }

    let mut serial = Pl011;
    serial.init();

    writeln!(
        serial,
        "[Kernel] {} GiB heap initialized",
        HEAP_SIZE / (1024 * 1024 * 1024)
    )
    .ok();
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

    for step in 0..10 {
        let seq_len = input_ids.len();
        let data = TensorData::new(input_ids.clone(), Shape::new([1, seq_len]));
        let input_tensor = Tensor::<Backend, 2, Int>::from_data(data, &device);

        let logits = model.forward(input_tensor);
        let last_logits = logits.slice([0..1, seq_len - 1..seq_len]);
        let next_token = last_logits.argmax(2).into_scalar() as i64;
        input_ids.push(next_token);

        writeln!(serial, "[step {}] token={}", step, next_token).ok();
    }

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