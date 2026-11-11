Experimental native Alpha toolchain — 2026-11-11

This is a private proof of concept, cross-built on the x86_64 workstation for
alphaev6-unknown-linux-gnu. It has now been installed through Portage on an ES40. Native C, C++ exception
handling, Cargo compilation, and Rust threads/atomics/file I/O passed there.
See ../portage/ES40-INSTALL.txt for physical-hardware evidence and commands.

Contents
  GCC 17 development C/C++ compiler and native libgccjit
  binutils 2.47.50.20261111 from unmodified upstream source
  rustc 1.102.0-dev with the GCC backend and Alpha standard library
  Cargo 1.101.0, including vendored libgit2, curl and OpenSSL
  cargo-alpha launcher and private libc/Rustix/Nix/OpenSSL build-helper sources

The installation prefix is /opt/rust-alpha-experimental. The tools are staged
under work/native-20261111/stage in the project. The full archive is
work/native-20261111/dist/alpha-native-toolchain-20261111-experimental.tar.xz.
The older alpha-native-gcc-binutils archive is a C-toolchain-only snapshot.

Manual extraction alternative (the ES40 used the Portage-owned installation)
  First check the machine's runtime and development headers against the Alpha
  glibc 2.41 sysroot used for this build. Transfer the archive and its .sha256
  file, verify with sha256sum -c, then extract under /:
    tar -xJf alpha-native-toolchain-20261111-experimental.tar.xz -C /
    export PATH=/opt/rust-alpha-experimental/bin:$PATH
    cargo-alpha new --vcs none hello-alpha
    cd hello-alpha
    cargo-alpha run

This uses a separate /opt prefix. It does not replace the system compiler.
The same directory may be extracted elsewhere; cargo-alpha resolves its own
prefix. Native system libc headers and libraries are required for C linking.
ALPHA_SYSROOT can select an alternative sysroot; the QEMU tests use this to
point to the existing Alpha SDK sysroot on the workstation.

cargo-alpha selects the native GCC linker, libgccjit, panic=abort and libc-only
Rustix mode. It passes Cargo patch settings for the bundled experimental
sources. Cargo may warn that some patches are unused; only dependencies in a
project's graph are used. An older lockfile may need, for example:
    cargo-alpha update -p libc --precise 0.2.191
cargo-alpha test enables panic-abort test support. rustdoc is not included, so
library tests should use --lib to avoid doctests. Ordinary cargo remains
available without these convenience settings.

Validated under QEMU
  Native assembler/linker smoke test and 57 Alpha linker regression tests.
  Native GCC/G++ compiling and running C and C++ exception tests.
  Native libgccjit: 16 compile contexts and 1,024 runtime checks.
  Alpha ELF Rust dylib metadata emitted and read by a second compilation.
  Native rustc compiling/running a std test with threads, collections and I/O.
  Native Cargo compiling and running a native procedural macro application.
  Six native C ABI probe binaries, with 18 successful checks.
  libc SYS_open, Rustix open/fstat and ioctl constants compared with C headers;
  Nix signal parsing and process checks for versions 0.30.1 and 0.31.3.
  Native Cargo version/help/new/metadata, Git creation and HTTPS registry search.
  See status.json and evidence/package-qemu.log for final staged-package checks.

Experimental limits
  No full native self-bootstrap or full Rust test suite. ES40 smoke tests passed.
  Panic unwinding is not enabled. Panics, including procedural-macro panics,
  and some compiler diagnostic failures can abort the process.
  Executables default to non-PIE because small Alpha PIE binaries still have
  an unresolved loader/linker problem in this environment.
  GCC emitted memory-access warnings during Cargo/backend compilation. Their
  causes are unresolved; passing smoke tests does not establish broad correctness.
  This is not an upstream support or readiness claim. The private patches and
  supporting text include LLM-generated work and remain for personal experimentation.

Build provenance
  binutils: 2de4446d9090b5773c863ab7bbdd95a5d4f81367
  GCC fork: e1c6e07b095d091177376a163da44039475e05e3
  Rust: 6f1c7ee0ec0ea64802f4b7a184b63b7efb645b3b
  GCC backend: f44a7221be06220f1aaa780faecc24781cba346c
  Cargo: c07ac2c4d12237effc4b94f4543769b421790045
  libc base: ccc2ff4aaab962f7fde7b8edda7379c045b1ca12 (0.2.191)
  GMP 6.3.0, MPFR 4.2.2, MPC 1.3.1; source SHA512 checks used GCC's manifest.

The baseline private patches are in rust_alpha_stuff/experimental/20261111/patches.
Additional native patches, scripts, lockfiles and evidence are in the adjacent
native-toolchain directory. The native additions include Alpha ELF object
selection, two bootstrap fixes (backend flags in the proc-macro probe and
installation of cross-built stage-zero std), SYS_open, dependency target
selection, and the non-PIE target default. Binutils has no local source changes.

The native compiler and backend were built by the isolated, patched x86_64
Rust compiler. Its cross-built Alpha std was installed from the authoritative
bootstrap artifact stamp; native-std-manifest.json records those exact files.
This avoids a redundant second workstation compiler build. It does not claim
the Alpha compiler has rebuilt itself.

Build/validation entry points in work/native-20261111
  build-binutils.sh, build-cross-cxx.py, build-deps.py, build-native-gcc.sh
  build-host-rust.sh, build-host-backend.sh, build-rust.sh, build-cargo.sh
  stage-rust.py, test-native-gcc.sh, test-native-rust.sh, test-native-abi.sh
  test-rustix.sh, test-native-cargo.sh, test-package.sh, export-native-evidence.py
These scripts use the prepared project sources and existing SDK; they are not
an installer for an empty machine. All build/cache/temp files remain on the
project disk. The previously validated cross-toolchain binaries are preserved.
