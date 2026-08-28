//! Build script for `ggml-sys`.
//!
//! Compiles a freestanding-capable `libggmlcpu.a` from the vendored
//! llama.cpp/ggml CPU backend (see `../ggml-src`, a trimmed snapshot of the
//! `ggml/` subtree).
//!
//! Two modes, selected by the cargo `TARGET`:
//!
//! * **Host** (anything not `*-unknown-none`): compile with the system
//!   clang/clang++ and the host SDK. Used by the `harness` crate to validate
//!   the ggml API and the single-kernel parity harness (T1/T4) on the dev box.
//!
//! * **Freestanding `aarch64-unknown-none`**: compile with the Arm GNU
//!   Toolchain (`aarch64-none-elf-gcc`/`g++`), which ships Newlib (C) and
//!   libstdc++ (C++). We compile in the toolchain's default (Newlib) mode —
//!   *not* `-ffreestanding` — because ggml's CPU backend needs the C++ standard
//!   library (`<vector>`, `<string>`, `<unordered_map>`, `std::mutex`, ...).
//!   The resulting object files are archived into `libggmlcpu.a`; the final
//!   link (done by the `kernel` crate) pulls in Newlib + libstdc++ and the
//!   minimal syscalls (`_sbrk`/`_write`/...) provided by `os-api`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// (relative-to ggml-src/ggml) source files -> compile as C (true) or C++ (false).
const GGML_SRCS: &[(&str, bool)] = &[
    ("src/ggml.c", true),
    ("src/ggml-alloc.c", true),
    ("src/ggml-quants.c", true),
    ("src/ggml-backend.cpp", false),
    ("src/ggml-backend-meta.cpp", false),
    ("src/gguf.cpp", false),
    ("src/ggml-cpu/ggml-cpu.c", true),
    ("src/ggml-cpu/ggml-cpu.cpp", false),
    ("src/ggml-cpu/repack.cpp", false),
    ("src/ggml-cpu/traits.cpp", false),
    ("src/ggml-cpu/vec.cpp", false),
    ("src/ggml-cpu/ops.cpp", false),
    ("src/ggml-cpu/binary-ops.cpp", false),
    ("src/ggml-cpu/unary-ops.cpp", false),
    ("src/ggml-cpu/quants.c", true),
    ("src/ggml-cpu/arch/arm/quants.c", true),
    ("src/ggml-cpu/arch/arm/repack.cpp", false),
];

fn run(cmd: &mut Command) {
    eprintln!("+ {:?}", cmd);
    let status = cmd.status().expect("failed to spawn compiler");
    if !status.success() {
        panic!("compiler failed: {:?}", cmd);
    }
}

/// Locate the Arm GNU Toolchain bin dir (`aarch64-none-elf-gcc` et al.).
///
/// Honours `$ARM_GNU_TOOLCHAIN` (a prefix containing `bin/`), else falls back
/// to searching `$PATH`. Returns the bin dir so callers can also find `ar`.
fn toolchain_bin() -> PathBuf {
    if let Ok(p) = env::var("ARM_GNU_TOOLCHAIN") {
        return PathBuf::from(p).join("bin");
    }
    // Fall back to PATH: find the compiler and derive its directory.
    let out = Command::new("bash")
        .arg("-lc")
        .arg("command -v aarch64-none-elf-gcc")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        });
    match out {
        Some(p) => PathBuf::from(p)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".")),
        None => panic!(
            "aarch64-none-elf-gcc not found. Install the Arm GNU Toolchain \
             (aarch64-none-elf) and set ARM_GNU_TOOLCHAIN, or put it on PATH."
        ),
    }
}

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ggml_root = manifest.join("..").join("ggml-src").join("ggml");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();
    let is_freestanding = target.ends_with("-none");

    let mut objs = Vec::new();

    if is_freestanding {
        let bin = toolchain_bin();
        let cc = bin.join("aarch64-none-elf-gcc");
        let cxx = bin.join("aarch64-none-elf-g++");
        let ar = bin.join("aarch64-none-elf-ar");

        // Single-threaded pthread shim (no real threads on the bare-metal
        // kernel; ggml runs with n_threads == 1).
        let shim = manifest.join("pthread-shim").join("pthread-stubs.c");
        let shim_obj = out.join("pthread-stubs.o");
        let mut cmd = Command::new(&cc);
        cmd.arg("-O2")
            .arg("-D_GNU_SOURCE")
            .arg("-D_POSIX_TIMERS")
            .arg("-D_POSIX_MONOTONIC_CLOCK")
            .arg("-D_POSIX_THREADS")
            .arg("-I")
            .arg(ggml_root.join("include"))
            .arg("-c")
            .arg(&shim)
            .arg("-o")
            .arg(&shim_obj);
        run(&mut cmd);
        objs.push(shim_obj);

        for (rel, is_c) in GGML_SRCS {
            let src = ggml_root.join(rel);
            let stem = rel.replace('/', "_");
            let obj = out.join(format!("{}.o", stem));
            let compiler = if *is_c { &cc } else { &cxx };
            let mut cmd = Command::new(compiler);
            cmd.arg("-O2")
                .arg("-fno-builtin")
                .arg("-D_GNU_SOURCE")
                .arg("-D_POSIX_TIMERS")
                .arg("-D_POSIX_MONOTONIC_CLOCK")
                .arg("-D_POSIX_THREADS")
                .arg("-I")
                .arg(ggml_root.join("include"))
                .arg("-I")
                .arg(&ggml_root.join("src"))
                .arg("-I")
                .arg(ggml_root.join("src").join("ggml-cpu"))
                .arg("-DGGML_VERSION=\"0.22.0\"")
                .arg("-DGGML_COMMIT=\"spike\"")
                .arg("-DGGML_BUILD_NUMBER=1");
            if *is_c {
                cmd.arg("-std=c11");
            } else {
                cmd.arg("-std=c++17");
            }
            cmd.arg("-c").arg(&src).arg("-o").arg(&obj);
            run(&mut cmd);
            objs.push(obj);
        }

        let lib = out.join("libggmlcpu.a");
        let mut cmd = Command::new(&ar);
        cmd.arg("rcs").arg(&lib);
        for o in &objs {
            cmd.arg(o);
        }
        run(&mut cmd);
    } else {
        let cc = env::var("CC").unwrap_or_else(|_| "clang".into());
        let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".into());

        for (rel, is_c) in GGML_SRCS {
            let src = ggml_root.join(rel);
            let stem = rel.replace('/', "_");
            let obj = out.join(format!("{}.o", stem));
            let compiler = if *is_c { &cc } else { &cxx };
            let mut cmd = Command::new(compiler);
            cmd.arg("-O1")
                .arg("-I")
                .arg(ggml_root.join("include"))
                .arg("-I")
                .arg(&ggml_root.join("src"))
                .arg("-I")
                .arg(ggml_root.join("src").join("ggml-cpu"))
                .arg("-DGGML_VERSION=\"0.22.0\"")
                .arg("-DGGML_COMMIT=\"spike\"")
                .arg("-DGGML_BUILD_NUMBER=1")
                .arg("-mmacosx-version-min=11.0");
            if *is_c {
                cmd.arg("-std=c11");
            } else {
                cmd.arg("-std=c++17").arg("-stdlib=libc++");
            }
            cmd.arg("-c").arg(&src).arg("-o").arg(&obj);
            run(&mut cmd);
            objs.push(obj);
        }

        let lib = out.join("libggmlcpu.a");
        let ar = env::var("AR").unwrap_or_else(|_| "ar".into());

        // Host-only no-op critical-section hooks (the target gets these from
        // `os-api`). Avoids a duplicate definition when linking the kernel.
        let shim = manifest.join("critical_section_shim.c");
        let shim_obj = out.join("critical_section_shim.o");
        let mut shim_cmd = Command::new(&cc);
        shim_cmd
            .arg("-O1")
            .arg("-std=c11")
            .arg("-I")
            .arg(ggml_root.join("include"))
            .arg("-c")
            .arg(&shim)
            .arg("-o")
            .arg(&shim_obj);
        run(&mut shim_cmd);
        objs.push(shim_obj);

        let mut cmd = Command::new(ar);
        cmd.arg("rcs").arg(&lib);
        for o in &objs {
            cmd.arg(o);
        }
        run(&mut cmd);

        // Host C++ runtime (macOS libc++ / libc++abi) for mutex, operator
        // delete, __cxa_guard_*, etc. that ggml-cpu references.
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=dylib=c++abi");
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=ggmlcpu");

    println!("cargo:rerun-if-changed={}", ggml_root.display());
    println!("cargo:rerun-if-env-changed=TARGET");
    if env::var("ARM_GNU_TOOLCHAIN").is_ok() {
        println!("cargo:rerun-if-env-changed=ARM_GNU_TOOLCHAIN");
    }
}
