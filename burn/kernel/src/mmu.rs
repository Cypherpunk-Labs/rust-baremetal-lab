// Minimal identity-mapped MMU setup.
//
// Under QEMU's HVF backend, guest exclusive-access instructions (ldaxrb/stxrb)
// to Device-attribute memory fault with a data abort. Real OSes work because
// they enable the MMU and run with Normal cacheable memory. This module
// enables a 1:1 stage-1 mapping (4 KiB granule, 4-level) with Normal
// Write-Back attributes so exclusive atomics execute natively.

use core::arch::asm;

// --- temporary debug UART (PL011 at 0x09000000) ---
const DBG_UART_DR: *mut u32 = 0x0900_0000 as *mut u32;
const DBG_UART_FR: *const u32 = (0x0900_0000 + 0x18) as *const u32;

fn dbg_putc(c: u8) {
    unsafe {
        while DBG_UART_FR.read_volatile() & (1 << 5) != 0 {}
        DBG_UART_DR.write_volatile(c as u32);
    }
}

fn dbg_str(s: &str) {
    for b in s.bytes() {
        if b == b'\n' {
            dbg_putc(b'\r');
        }
        dbg_putc(b);
    }
}

fn dbg_u64(v: u64) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 18];
    buf[0] = b'0';
    buf[1] = b'x';
    for i in 0..16 {
        buf[2 + i] = HEX[((v >> (60 - i * 4)) & 0xf) as usize];
    }
    for b in buf {
        dbg_putc(b);
    }
}
// --- end debug ---

#[repr(C, align(4096))]
struct PageTable([u64; 512]);

static mut L0: PageTable = PageTable([0; 512]);
static mut L1: PageTable = PageTable([0; 512]);

const ATTR_NORMAL: u64 = 0; // MAIR_EL1 field 0: Normal WB, RA, WA
const ATTR_DEVICE: u64 = 1; // MAIR_EL1 field 1: Device-nGnRE

// Block descriptor lower bits: type=0b01 (block), AF, SH=11, AP=00 (EL1 RW only).
const BLOCK_FLAGS: u64 = 0b01 | (1 << 10) | (3 << 8) | (0b00 << 6);
const TABLE: u64 = 0b11;

unsafe fn desc_block(phys: usize, attr: u64) -> u64 {
    (phys as u64 & !0x1f) | BLOCK_FLAGS | (attr << 4)
}

// UART (and GIC) live at 0x0800_0000..0x0a00_0000; map Device.
    fn is_uart(addr: usize) -> bool {
        addr >= 0x0800_0000 && addr < 0x0a00_0000
    }

    // 2-level identity map: L0[0] -> L1, L1[i] = 1 GiB block.
    fn fill_l1(table: *mut PageTable) {
        let t = &raw const *table;
        for i in 0..512usize {
            let addr = i * (1usize << 30);
            if addr >= 0x4_0000_0000 {
                unsafe { (*t.cast_mut()).0[i] = 0 };
                continue;
            }
            let attr = if is_uart(addr) { ATTR_DEVICE } else { ATTR_NORMAL };
            unsafe { (*t.cast_mut()).0[i] = desc_block(addr, attr) };
        }
    }

    /// Enable an identity (1:1) page table covering the first 4 GiB.
    /// Must be called once, from EL1, with the MMU still off.
    pub unsafe fn init(_serial: &mut impl core::fmt::Write) {
        unsafe {
            let l0 = core::ptr::addr_of!(L0) as usize;
            let l1 = core::ptr::addr_of!(L1) as usize;

            fill_l1(&raw mut L1);
            L0.0[0] = l1 as u64 | TABLE;

        // MAIR_EL1: field0 = Normal WB (0xff), field1 = Device-nGnRE (0x04).
        asm!("msr mair_el1, {0}", in(reg) 0xffu64 | (0x04u64 << 8));
        // TCR_EL1: T0SZ=16/T1SZ=16 (48-bit VA), TG0=4KiB, WB cache, inner shareable, 48-bit IPS.
        let tcr = (16u64) | (0b00u64 << 14) | (0b01u64 << 8) | (0b01u64 << 10) | (0b11u64 << 12) | (0b101u64 << 32) | (16u64 << 16);
        asm!("msr tcr_el1, {0}", in(reg) tcr);
        asm!("msr ttbr0_el1, {0}", in(reg) l0 as u64);
        asm!("msr ttbr1_el1, {0}", in(reg) l0 as u64);
        asm!("isb");
        asm!("tlbi vmalle1");
        asm!("dsb ish");
        asm!("isb");

        dbg_str("mmu: l0=");
        dbg_u64(l0 as u64);
        dbg_str(" l1=");
        dbg_u64(l1 as u64);
        dbg_str("\nmmu: L0[0]=");
        dbg_u64(L0.0[0]);
        dbg_str(" L1[0]=");
        dbg_u64(L1.0[0]);
        dbg_str(" L1[2]=");
        dbg_u64(L1.0[2]);
        dbg_str(" L1[0x80]=");
        dbg_u64(L1.0[0x80]);
        dbg_str("\n");

        // read back sysregs to confirm the writes took
        let mut r: u64;
        asm!("mrs {0}, mair_el1", out(reg) r);
        dbg_str("mmu: mair_el1=");
        dbg_u64(r);
        asm!("mrs {0}, tcr_el1", out(reg) r);
        dbg_str(" tcr_el1=");
        dbg_u64(r);
        asm!("mrs {0}, ttbr0_el1", out(reg) r);
        dbg_str(" ttbr0_el1=");
        dbg_u64(r);
        asm!("mrs {0}, sctlr_el1", out(reg) r);
        dbg_str(" sctlr_el1=");
        dbg_u64(r);
        dbg_str("\n");

        // Software walk of VA 0x400a0000 (kernel region) to mimic the CPU.
        let va: u64 = 0x400a_0000;
        let l0i = (va >> 39) & 0x1ff;
        let l1i = (va >> 30) & 0x1ff;
        let l0e = L0.0[l0i as usize];
        let l1e = if l0e & 0b11 == 0b11 { L1.0[l1i as usize] } else { 0 };
        dbg_str("mmu: walk L0[");
        dbg_u64(l0i);
        dbg_str("]=");
        dbg_u64(l0e);
        dbg_str(" L1[");
        dbg_u64(l1i);
        dbg_str("]=");
        dbg_u64(l1e);
        dbg_str(" block=");
        dbg_u64(l1e & !0x7f);
        dbg_str("\n");

        // Enable MMU (M) only, no caches, to isolate.
        let mut sctlr: u64;
        asm!("mrs {0}, sctlr_el1", out(reg) sctlr);
        sctlr |= 1 << 0;
        asm!("msr sctlr_el1, {0}", in(reg) sctlr);
        asm!("isb");
        asm!("tlbi vmalle1");
        asm!("dsb ish");
        asm!("isb");
        dbg_str("mmu: enabled\n");
    }
}