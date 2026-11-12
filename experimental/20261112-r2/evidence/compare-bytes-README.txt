Minimized Alpha rustc_codegen_gcc memcmp ICE, 2026-11-12

Three-line reproducer: minimal.rs
No external crates, Cargo, locale data, unsafe execution or nightly features
needed to trigger the compilation failure. Compile as a library to retain the
public function. On ES40 with original installed experimental compiler:

/opt/rust-alpha-experimental/bin/rustc minimal.rs --crate-type=rlib \
  -Copt-level=3 -Cpanic=abort -o libminimal.rlib

Expected: successful compilation. Actual: compiler SIGABRT, libgccjit rejects
memcmp argument3: expected size_t, got signed aligned __int64_t.
The unpatched cross backend reproduces the same error (exit101 rather than
native SIGABRT/134). Cross optimization matrix: opt0/opt1 PASS, opt2/opt3 ICE.

Native candidate validation (same source/options; RUSTC_BOOTSTRAP=1 plus
-Zcodegen-backend=/root/rust-alpha-overnight-20261111/compiler-probe/backend.so):
compilation PASS. Linked runtime-minimal.rs against fixed rlib; equal, unequal,
empty, longer and high-byte C strings all give expected comparison results.
See minimal-old.log, minimal-fixed.log, runtime-minimal.log.
No installed toolchain replacement or release repack.

Reduction:
1. Original locale0.2.2 direct rustc invocation fails (no Cargo required).
2. Removing macos module still fails.
3. Removing Linux get_iconv's codeset comparison lets locale compile.
4. Extracting CStr::from_ptr -> unchecked str -> UTF-8 comparison still fails.
5. Remove locale FFI call, UTF-8 conversion and all crate dependencies:
   CStr::from_ptr(p).to_bytes() == b"x".as_slice() still fails.

Controls / relevant limits:
- Comparing to b"x" directly (array reference), rather than its slice, passes.
- Ordinary slice arguments and CStr reference arguments tested here pass.
- An explicit strlen -> from_raw_parts version passes.
- CStr internally uses UsizeNoHighBitMinusOne and transmutes back to usize.
  This is a plausible route to the signed temporary; direct pattern-type and
  NonZero/transmute probes did NOT reproduce. Do not claim this alone proves
  the exact MIR transformation responsible.
- The confirmed failure boundary is compare_bytes calling memcmp without
  converting its argument to C size_t. The candidate adds that conversion,
  consistent with raw_eq and memcpy/memmove/memset.

The bug is demonstrated on the experimental Alpha target with Rust1.102.0-dev
and GCC17 libgccjit. Other architectures/toolchain versions are not assessed.
