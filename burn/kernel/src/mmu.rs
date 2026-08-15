// Minimal identity-mapped MMU setup.
//
// Under QEMU's HVF backend, exclusive-access instructions (ldaxrb/stxrb) to
// Device-attribute memory fault with a data abort (EC=0x25, DFSC=0x35) because
// HVF has no emulation for them. Real OSes work because they enable the MMU and
// run with Normal cacheable memory. This module enables a 1:1 stage-1 mapping
// (4 KiB granule) with Normal Write-Back attributes so exclusive atomics
// execute natively under HVF.
//
// Table layout: L0[0] -> L1; L1[i] = 1 GiB block for the first 16 GiB.
// AP=0b00 (EL1 RW only) is required: a page writable at EL0 is execute-never
// at EL1 (QEMU get_S1prot), so any AP != 0b00 aborts the first instruction
// fetch after SCTLR_EL1.M is set (Permission fault, IFSC=0x0d).

use core::arch::asm;

#[repr(C, align(4096))]
struct PageTable([u64; 512]);

static mut L0: PageTable = PageTable([0; 512]);
static mut L1: PageTable = PageTable([0; 512]);

// Block descriptor lower bits: type=0b01 (block), AF, SH=11, AP=00 (EL1 RW only).
const BLOCK_FLAGS: u64 = 0b01 | (1 << 10) | (3 << 8) | (0b00 << 6);
const TABLE: u64 = 0b11;

unsafe fn desc_block(phys: usize) -> u64 {
    (phys as u64 & !0x1f) | BLOCK_FLAGS
}

// 2-level identity map: L0[0] -> L1, L1[i] = 1 GiB block (first 16 GiB).
unsafe fn fill_l1() {
    for i in 0..512usize {
        let addr = i * (1usize << 30);
        L1.0[i] = if addr >= 0x4_0000_0000 { 0 } else { desc_block(addr) };
    }
    L0.0[0] = core::ptr::addr_of!(L1) as u64 | TABLE;
}

/// Enable a 1:1 identity page table over the first 16 GiB. Must be called
/// once, from EL1, with the MMU still off.
pub unsafe fn init() {
    unsafe {
        let l0 = core::ptr::addr_of!(L0) as usize;

        fill_l1();

        // MAIR_EL1: field0 = Normal WB (0xff).
        asm!("msr mair_el1, {0}", in(reg) 0xffu64);
        // TCR_EL1: T0SZ/T1SZ=16 (48-bit VA), TG0=4KiB, WB cache,
        // inner shareable, 48-bit IPS. TTBR1 mirrors TTBR0 so any address
        // selects a valid table.
        let tcr = (16u64)
            | (0b00u64 << 14)
            | (0b01u64 << 8)
            | (0b01u64 << 10)
            | (0b11u64 << 12)
            | (0b101u64 << 32)
            | (16u64 << 16);
        asm!("msr tcr_el1, {0}", in(reg) tcr);
        asm!("msr ttbr0_el1, {0}", in(reg) l0 as u64);
        asm!("msr ttbr1_el1, {0}", in(reg) l0 as u64);
        asm!("isb");
        asm!("tlbi vmalle1");
        asm!("dsb ish");
        asm!("isb");

        // Enable MMU (M) only; caches stay off (irrelevant under QEMU).
        let mut sctlr: u64;
        asm!("mrs {0}, sctlr_el1", out(reg) sctlr);
        sctlr |= 1 << 0;
        asm!("msr sctlr_el1, {0}", in(reg) sctlr);
        asm!("isb");
        asm!("tlbi vmalle1");
        asm!("dsb ish");
        asm!("isb");
    }
}