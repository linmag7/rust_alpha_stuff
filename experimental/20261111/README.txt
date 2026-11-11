Historical cross-build checkpoint. For the later native ES40 toolchain and
Portage installation, start at the repository-root README.md,
native-toolchain/README.txt and portage/ES40-INSTALL.txt.

Private Alpha Rust proof of concept — 2026-11-11

This work includes LLM-generated implementation. It is a private experiment,
not a submission-ready Rust contribution. Component contribution policies must
be considered separately before sharing or submitting any of this work.

Compiler: Rust 1.102.0-dev, built through stage 2 from the fresh Rust checkout.
Code generation: rustc_codegen_gcc with the fixed GCC 17 Alpha libgccjit.
Linking/runtime: upstream binutils 2.47.50.20261111, SDK GCC 16 C driver,
and the glibc 2.41 sysroot. See binutils-upgrade/README.txt for exact revision,
integration, test results and the remaining small release-PIE limitation.
Target: alphaev6-unknown-linux-gnu (EV6 or newer), panic=abort.
The earlier packaged SDK has not been replaced.

From the project directory:
  work/experimental-20261111/cargo-alpha build --manifest-path /path/to/Cargo.toml
  work/experimental-20261111/cargo-alpha run --manifest-path /path/to/Cargo.toml
  work/experimental-20261111/cargo-alpha test --manifest-path /path/to/Cargo.toml
Add --release for an optimized build. The default runner is QEMU.
For the included std demonstration:
  work/experimental-20261111/cargo-alpha run --manifest-path work/experimental-20261111/probes/std_probe/Cargo.toml

Build helpers and validation:
  /media/lindholm/545a6254-41a9-48f3-a207-cbea64457d45/gcc-alpha-rust/work/experimental-20261111/build-rust.sh
  /media/lindholm/545a6254-41a9-48f3-a207-cbea64457d45/gcc-alpha-rust/work/experimental-20261111/build-backend.sh
  python3 /media/lindholm/545a6254-41a9-48f3-a207-cbea64457d45/gcc-alpha-rust/work/experimental-20261111/validate.py
Build Rust before its backend; rebuilding the backend refreshes the toolchain
content ID, which separates Cargo caches after compiler/backend/GCC, binutils
or wrapper changes.
All new build output, caches and temporary files are on the project disk.

Limits:
  This is a cross compiler hosted on x86_64 Linux, not a native Alpha compiler.
  Rust C-variadic function definitions are disabled pending Alpha va_list support.
  f128 is not reported as reliable because GCC Alpha comparisons currently ICE.
  x87 f80 is unsupported by this experimental backend adaptation.
  LTO and Cargo incremental compilation are disabled in the wrapper.
  Panic unwinding and doctests are not enabled by this wrapper.
  The new Rust/backend combination has not yet been tested on the ES40, which
  was unreachable during this session. QEMU validation is recorded separately.

Results, exact revisions, provenance and patch hashes are in status.json and
validation.json alongside this file. Detailed logs remain under:
  /media/lindholm/545a6254-41a9-48f3-a207-cbea64457d45/gcc-alpha-rust/work/experimental-20261111/logs
