#![no_std]
#![no_main]

use core::arch::global_asm;
use core::fmt::{self, Write};
use core::panic::PanicInfo;

const UART0: *mut u8 = 0x0900_0000 as *mut u8;

struct Uart;

impl Write for Uart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            unsafe {
                core::ptr::write_volatile(UART0, byte);
            }
        }
        Ok(())
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

global_asm!(
    ".global _start",
    "_start:",
    "ldr x30, =0x44000000",
    "mov sp, x30",
    "bl rust_main",
    "b ."
);

#[unsafe(no_mangle)]
pub extern "C" fn rust_main() -> ! {
    let mut uart = Uart;
    let _ = writeln!(uart, "Hello baremetal world from ARM64 on MacOS!");
    loop {}
}