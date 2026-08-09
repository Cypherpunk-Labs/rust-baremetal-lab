## Phase 2: The ARM64 Kernel

ARM64 has a built-in baremetal target, but requires a custom linker script and inline assembly to initialize the stack and map memory correctly for the QEMU `virt` machine.

**1. Initialize the Project**

```bash
cargo new os_arm --bin
cd os_arm
```

**2. Create the Linker Script (`linker.ld`)**
Create this file in the `os_arm` root directory to tell the compiler where QEMU physically loads RAM.

```ld
ENTRY(_start)

SECTIONS {
    . = 0x40080000;
    .text : { *(.text._start) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) }
}

```

**3. Configure Cargo (`.cargo/config.toml`)**
Create a `.cargo` directory and `config.toml` file to enforce the linker script.

```toml
[target.aarch64-unknown-none]
rustflags = ["-C", "link-arg=-Tlinker.ld"]

```

**4. Write the Kernel (`src/main.rs`)**

```rust
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

```

**5. Build and Boot**

```bash
cargo build --target aarch64-unknown-none
qemu-system-aarch64 -machine virt -cpu cortex-a53 -nographic -kernel target/aarch64-unknown-none/debug/os_arm

```

*Note: To exit QEMU in the terminal, press `Ctrl-a` then `x`.*

[x] MK Tested 09/08/2026

[Back](README.md)