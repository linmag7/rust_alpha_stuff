Upstream binutils for the private Alpha Rust experiment — 2026-11-11

The experimental cargo-alpha now uses unmodified upstream binutils
2.47.50.20261111, revision 2de4446d9090b5773c863ab7bbdd95a5d4f81367,
fetched from https://sourceware.org/git/binutils-gdb.git.
No Matt Turner patches were copied or reimplemented. The existing packaged
SDK remains available; its GCC 16 C compiler and glibc 2.41 sysroot are reused.

Use from the project directory:
  work/experimental-20261111/cargo-alpha build --manifest-path /path/to/Cargo.toml
  work/experimental-20261111/cargo-alpha run --manifest-path /path/to/Cargo.toml
  work/experimental-20261111/cargo-alpha test --manifest-path /path/to/Cargo.toml
Add --release for optimized builds. QEMU remains the default runner.

Integration:
  work/binutils-20261111/install contains the new cross binutils.
  work/binutils-20261111/alpha-cc selects its assembler/linker with -B.
  The GCC 17/libgccjit build was reconfigured with absolute --with-as and
  --with-ld paths to the new tools, then rebuilt. GCC source was not changed.
  strace evidence confirms both the JIT assembler and C assembler/linker paths.
  Cargo's toolchain content ID now also hashes binutils and the wrappers:
  ddb0c065314fd2fa. Existing target caches remain separate.

Validation:
  34 normal Rust validation commands passed: debug/release std, C ABI, libc,
  inline assembly, Serde/Clap/Regex/procedural macros/C compilation, Cargo tests.
  The libc constant comparison passed all 2367 constants in each profile.
  All 18 normal ABI/libc probe executions passed again after PIE diagnostics.
  A small C PIE also passed under QEMU.

  Upstream suites: ld 1398 PASS, 16 FAIL, 1 XPASS, 30 XFAIL, 378 UNSUPPORTED;
  gas 331 PASS, 5 FAIL, 19 UNSUPPORTED;
  binutils 272 PASS, 1 XFAIL, 11 UNSUPPORTED.
  Every current ld FAIL/XPASS was also observed using the old 2.45 tools.
  All five gas failures reproduce with both assemblers (.set alias syntax).
  This is not a clean full-suite result. The old-tools comparison was stopped
  after matching the current failures, while an unrelated old linker test
  remained running for over 90 seconds. Its isolated plugin test setup also
  lacked a built test plugin; the baseline is not a full-suite certification.

  Matt's upstream regressions pass: got-realloc, got-merge-gprel, relax-gpdisp.
  On the same cases, old 2.45 aborts on GOT reallocation, segfaults on the
  large-GOT merge case, and incorrectly removes the caller's GP reload.

Remaining limits:
  The nine debug ABI/libc PIE probes pass. All nine release PIE probes still
  fail with DT_JMPREL=0 and loader exit 127. Keep their existing -no-pie
  workaround. Normal std and compatibility application validation passed.
  IFUNC link tests do not establish IFUNC runtime support in glibc 2.41.
  The ES40 did not respond to the SSH TCP connection attempt; this upgrade
  has been validated under QEMU only.
  Existing panic=abort, disabled LTO/incremental, and f128 limitations remain.

Reproduction (all workspace and temporary data stay on the project disk):
  Sources: work/binutils-20261111/src, checked out at the exact revision above.
  Build/install binutils: work/binutils-20261111/build.sh
  Reconfigure/rebuild JIT: python3 work/binutils-20261111/rebuild-jit.py
  Refresh cache identity: python3 work/experimental-20261111/refresh-toolchain-id.py
  Upstream suites: work/binutils-20261111/check.sh
    This exits nonzero for the recorded suite failures; inspect the .sum files.
  Matt's regression comparison: python3 work/binutils-20261111/compare-linkers.py
  Rust validation: python3 work/experimental-20261111/validate.py
  PIE diagnostic: python3 work/binutils-20261111/validate-pie.py
  Restore normal probes afterward:
    ALPHA_RESTORE_PROBES=1 python3 work/binutils-20261111/validate-pie.py

Full logs and baseline snapshots: work/binutils-20261111.
This directory contains copied evidence, checksums, exact provenance and
script snapshots. Run the working copies at the paths above: script snapshots
retain paths relative to their original workspace locations.
The local integration helpers include LLM-generated code; this remains a
private experiment. No external submission or publication was performed.
