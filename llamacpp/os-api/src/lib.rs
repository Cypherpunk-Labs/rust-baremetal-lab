#![no_std]
extern crate alloc;

// Only provide the shim on the bare-metal target. On the host, system libc
// already supplies all of these.
#[cfg(target_os = "none")]
mod imp {
    use core::ffi::{c_char, c_int, c_long, c_void};

    // ---------------------------------------------------------------------
    // Heap (shared by Newlib's malloc and the Rust global allocator).
    // Must match the region the kernel hands to the MMU / QEMU RAM size.
    // ---------------------------------------------------------------------
    const HEAP_START: usize = 0x7000_0000;
    const HEAP_SIZE: usize = 0x8000_0000; // 2 GiB

    static mut SBRK_CUR: usize = HEAP_START;

    #[no_mangle]
    pub extern "C" fn _sbrk(incr: isize) -> *mut c_void {
        unsafe {
            let cur = SBRK_CUR;
            let new = (cur as isize + incr) as usize;
            if new < cur || new > HEAP_START + HEAP_SIZE {
                return usize::MAX as *mut c_void; // (void*)-1
            }
            SBRK_CUR = new;
            cur as *mut c_void
        }
    }

    // ---------------------------------------------------------------------
    // UART sink: the kernel installs its PL011 putc; Newlib's printf/fprintf
    // reach the UART through `_write`.
    // ---------------------------------------------------------------------
    type Putc = unsafe extern "C" fn(c_char);
    static mut UART_PUTC: Option<Putc> = None;

    /// Host-side callers and the kernel both use this to install the UART sink.
    #[no_mangle]
    pub extern "C" fn install_uart_putchar(f: unsafe extern "C" fn(c_char)) {
        unsafe { UART_PUTC = Some(f); }
    }

    #[no_mangle]
    pub unsafe extern "C" fn _write(_fd: c_int, buf: *const u8, cnt: usize) -> isize {
        if let Some(putc) = UART_PUTC {
            for i in 0..cnt {
                putc(*buf.add(i) as c_char);
            }
        }
        cnt as isize
    }

    #[no_mangle]
    pub extern "C" fn _read(_fd: c_int, _buf: *mut u8, _cnt: usize) -> isize {
        0
    }

    #[no_mangle]
    pub extern "C" fn _close(_fd: c_int) -> c_int {
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn _fstat(_fd: c_int, st: *mut c_void) -> c_int {
        core::ptr::write_bytes(st as *mut u8, 0, 128);
        0
    }

    #[no_mangle]
    pub extern "C" fn _isatty(fd: c_int) -> c_int {
        if fd <= 2 { 1 } else { 0 }
    }

    #[no_mangle]
    pub extern "C" fn _lseek(_fd: c_int, _off: isize, _wh: c_int) -> isize {
        0
    }

    #[no_mangle]
    pub extern "C" fn _clock_gettime(_clk: c_int, tp: *mut c_void) -> c_int {
        // struct timespec { time_t tv_sec; long tv_nsec; } -> report epoch 0.
        unsafe {
            let p = tp as *mut u64;
            *p = 0;
            *p.add(1) = 0;
        }
        0
    }

    #[no_mangle]
    pub extern "C" fn _getpid() -> c_int {
        1
    }

    #[no_mangle]
    pub extern "C" fn _kill(_pid: c_int, _sig: c_int) -> c_int {
        -1
    }

    #[no_mangle]
    pub extern "C" fn _times(_buf: *mut c_void) -> c_int {
        0
    }

    // ggml/Newlib call these directly; Newlib's prebuilt libc.a in this bare
    // config does not export them, so we provide them here.
    #[no_mangle]
    pub extern "C" fn clock_gettime(clk: c_int, tp: *mut c_void) -> c_int {
        _clock_gettime(clk, tp)
    }

    #[no_mangle]
    pub extern "C" fn clock() -> u32 {
        0
    }

    #[no_mangle]
    pub extern "C" fn sysconf(_name: c_int) -> c_long {
        1
    }

    // ggml_aligned_malloc uses posix_memalign. Back it with Newlib's memalign,
    // whose result is safe to pass to free().
    extern "C" {
        fn memalign(align: usize, size: usize) -> *mut c_void;
    }

    #[no_mangle]
    pub unsafe extern "C" fn posix_memalign(
        memptr: *mut *mut c_void,
        align: usize,
        size: usize,
    ) -> c_int {
        let p = memalign(align, size);
        if p.is_null() {
            return 12; // ENOMEM
        }
        *memptr = p;
        0
    }

    // File/stat syscalls: nothing is opened on the bare-metal target, so these
    // are stubs. They only need to exist so Newlib's stdio links; callers that
    // actually open files (e.g. GGUF-from-disk) are not used on-device (we load
    // the model from an in-memory buffer).
    #[no_mangle]
    pub extern "C" fn _open(_path: *const c_char, _flags: c_int, _mode: c_int) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn _stat(_path: *const c_char, _st: *mut c_void) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn _access(_path: *const c_char, _mode: c_int) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn _unlink(_path: *const c_char) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn _gettimeofday(_tp: *mut c_void, _tz: *mut c_void) -> c_int {
        0
    }

    // newlib assert() / abort() path.
    #[no_mangle]
    pub extern "C" fn __assert_func(
        _file: *const c_char,
        _line: c_int,
        _func: *const c_char,
        _expr: *const c_char,
    ) -> ! {
        loop {
            core::hint::spin_loop();
        }
    }

    #[no_mangle]
    pub extern "C" fn _exit(_code: c_int) -> ! {
        loop {
            core::hint::spin_loop();
        }
    }

    // ggml-threading.cpp is not compiled (its `std::mutex` needs a threaded
    // libstdc++; the kernel is single-threaded). Provide the critical-section
    // hooks it would have defined as trivial no-ops.
    #[no_mangle]
    pub extern "C" fn ggml_critical_section_start() {}
    #[no_mangle]
    pub extern "C" fn ggml_critical_section_end() {}

    // Normally provided by crt0 (disabled via -nostartfiles). Referenced by
    // libstdc++ for C++ static-duration destructor registration.
    #[no_mangle]
    pub static __dso_handle: u8 = 0;
}

/// Host-side callers and the kernel both use this to install the UART sink.
#[cfg(target_os = "none")]
pub use imp::install_uart_putchar;
