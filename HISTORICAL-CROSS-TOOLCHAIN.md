# Rust on Alpha Linux

An experimental Linux/Alpha cross-toolchain using patched Rust 1.94-dev,
`rustc_codegen_gcc`, and an x86_64-hosted Alpha GCC/libgccjit. It now supports
`std` with `panic=abort`, in addition to the preserved `no_std` + `alloc` baseline.
Builds and caches stay under `../work/` on the project disk.


## Alpha SDK

The packaged SDK adds `cargo alpha build`, `check`, `run`, and `test` around the
validated compiler. It supplies the Alpha target, libc patch, C compiler,
standard-library build flags, and QEMU runner without application manifest edits.
Downloads are enabled by default; `--offline` remains available.

From the project directory:

```sh
export PATH="$PWD/work/sdk/release/alpha-sdk/bin:$PATH"
cargo alpha doctor --verify
cargo alpha new --vcs none work/my-alpha-app
cd work/my-alpha-app
cargo alpha run
cargo alpha build --release
cargo alpha test
cargo alpha run --runner es40 --release
```

The installed SDK has the ES40 profile configured for this workstation. The
archive includes `config.example.json`; copy it to `config.json` after extracting
elsewhere. SSH uses the codex account's existing key and prompts for that local
account's password. No credentials are packaged.

The SDK is assembled by `sdk/package.py`; its guide is [sdk/USAGE.txt](sdk/USAGE.txt).
It includes the ES40's glibc 2.41 sysroot and patched libc 0.2.191, while retaining
the validated Rust/GCC 16 compiler pair. The newer compiler rebase is separate.
`sdk/validate.py` exercises independent Cargo projects, pinned third-party crates,
procedural macros, C build scripts, failure reporting, and the existing ABI/libc
regression probes. Unit and integration tests support expected panics through
the aborting-panic test harness; doctests and unwinding remain unsupported.


## Validated on 2026-11-11

All debug and release probes pass under QEMU with the glibc 2.42 sysroot and on
the ES40 (EV67, glibc 2.41, 8192-byte pages).

| Coverage | QEMU debug/release | ES40 debug/release |
| --- | --- | --- |
| Scalars, stack arguments, aggregate arguments/returns, C callbacks | PASS | PASS |
| Former 24-byte argument/callback failures; 64-byte and register/stack split arguments | PASS | PASS |
| Core/alloc, arithmetic, inline asm, over-aligned allocation | PASS | PASS |
| 315 libc layout/type checks; 2,370 cross-header constants | PASS | PASS |
| Memory mapping, signals, errno, terminal speed helpers | PASS | PASS |
| std: files/metadata, hashing, formatting, threads/TLS/atomics/locks, clocks, sockets, child processes | PASS | PASS |

Native GCC on the ES40 independently confirmed all 315 layout values and all
2,367 constants available in its headers, with zero mismatches. ELF inspection
confirms that each of the five terminal speed helpers imports `GLIBC_2.0`.

Results: `../work/logs/validated-qemu.log`,
`../work/logs/es40-abi-std-20261111.log`, and
`../work/logs/termios-symbol-versions.txt`. Artifact hashes are in
`../work/logs/validated-toolchain-sha256.txt` and the bundle's `SHA256SUMS`.
The current bundle is `../work/es40-abi-std.tar.gz`; its ES40 copy and native
compiler results are in `/root/rust-alpha-abi-std-20261111/`.
No passwords or SSH keys are stored in the repository or bundle.

## Build and run

From this repository, with the prepared toolchain:

```sh
# Cross-build an application using the patched std and local libc fork.
./scripts/alpha-std-build.sh "$PWD/std_probe/Cargo.toml" --release
source ./scripts/alpha-fixed-env.sh
./scripts/alpha-run.sh "$ALPHA_STD_TARGET_DIR/alphaev6-unknown-linux-gnu/release/alpha_std_probe"

# Build and validate debug and release: ABI, libc, allocation, inline asm, std.
./scripts/alpha-validate.sh > ../work/logs/validated-qemu.log 2>&1

# Produce a bundle to unpack on an Alpha machine and run with:
# bash ./run-tests-native.sh
./scripts/alpha-bundle.sh
```

For another application, pass its Cargo.toml to `alpha-std-build.sh`. Builds use
Cargo's unstable `-Zbuild-std` and the local libc patch. Dependencies must be
available in `work/cargo-home`, since these scripts build offline. This does not
yet provide a native Alpha rustc or Cargo executable.

`alpha-fixed-env.sh` selects the isolated compiler/backend and the Unix target
spec. It keeps output directories separate using `work/toolchain-id`, a content
hash of the compiler and backend; local rustc rebuilds otherwise report identical
version strings and can cause Cargo to reuse stale artifacts. Explicit
`ALPHA_TARGET_DIR` and `ALPHA_STD_TARGET_DIR` overrides remain available.

The older `alpha-build.sh`, `alpha-run.sh`, and `alpha-check.sh` retain their
installed-toolchain defaults for reproducing the original no_std baseline.
Source `alpha-fixed-env.sh` before using those with the corrected compiler.

## Changes

- **Alpha C ABI:** ordinary aggregate arguments stay by value at every size;
  GCC assigns the initial argument words to registers and the remainder to the
  stack. Aggregate results use the OSF/Linux indirect return convention. Both
  signed and unsigned 32-bit scalar argument items are sign-extended.
- **Bootstrap:** prevent a local rebuild from copying installed rustc-dev
  sources through the `rustc-src` symlink into the working source tree. This
  previously overwrote the classifier edit before it could be compiled.
- **libc:** a local fork of libc 0.2.178 with a separate Alpha GNU module,
  Alpha integer aliases, records, signal and pthread structures, ioctl encoding,
  and architecture-specific constants. Alpha cannot reuse a generic LP64 module:
  for example, `ino_t` and `blkcnt_t` are 32-bit here.
- **std glue:** Alpha Linux raw types, allocator minimum alignment, and unwind
  structure/register definitions required to compile std. Runtime unwinding is
  still unvalidated; the supported configuration uses aborting panics.
- **Symbol versions:** the backend lowers versioned Linux function imports via
  assembler `.symver` aliases, since libgccjit rejects `@` in function names.
  This permits the Alpha libc baud helpers to bind to `GLIBC_2.0`.

The target spec uses GNU archive grouping to resolve the circular references
between std and its panic runtime with the saved linker. Small no_std ABI/libc
probes also use a local `-no-pie` workaround: the saved binutils otherwise emits
an invalid `DT_JMPREL` in some small PIEs. The std integration probe runs as PIE.

## libc derivation and validation

The bindings come from the Alpha cross sysroot's headers, not host layouts.
`libc_probe/generate_types.py` extracts typedefs and record fields from GCC DWARF.
`generate_constants.py` takes names from libc modules and evaluates their values
with the Alpha C compiler. The checked-in C probes independently supply expected
sizes, alignments, offsets, signedness and constants for the Rust probes.

Normal validation requires the cross compiler and QEMU, not pyelftools. To
regenerate the bindings, install pyelftools 0.32 in the project workspace (the
prepared copy is `work/pyelftools-0.32`) and run
`./scripts/alpha-regenerate-libc.sh`. Review the resulting diff before using it.
The tests cover 315 layout/type properties and 2,370 exported constants; they are
not a complete audit of every libc API. Kernel-only `termios2` is outside the
glibc record-layout suite. `TCP_TX_DELAY` is excluded because libc 0.2.178 does
not export it (the name collector also finds its commented declaration).

The cross sysroot has glibc 2.42; the ES40 has 2.41. Those versions disagree on
baud-rate macro values. This fork deliberately uses the legacy encoded constants
and pins all five `cf*speed` functions to Alpha's `GLIBC_2.0` versions. The C probe
selects the corresponding `__B*` constants when newer headers provide them.
Runtime tests exercise all five functions and check the encoded control flags.
See the [glibc baud-rate compatibility discussion](https://github.com/rust-lang/libc/issues/4692).
The native runner also compiles C probes with the ES40's own headers and compares
them with the cross sysroot, in addition to executing the Rust binaries.

## Sources and rebuild

```sh
./scripts/alpha-rebuild-toolchain.sh > ../work/logs/fixed-toolchain-build.log 2>&1
./scripts/alpha-validate.sh > ../work/logs/validated-qemu.log 2>&1
```

This rebuild uses the preserved source/toolchain inputs already on these disks.
It is not a bootstrap from an arbitrary stock Rust release. Existing isolated
source trees are retained so local changes are not overwritten; on a fresh
workspace the script copies the preserved sources and applies the patches.

| Input | Location/version |
| --- | --- |
| Preserved Rust sources | `../rust`, HEAD `7249ed41994`, initial Alpha support |
| Stage0 Rust/Cargo | `/home/lindholm/toolchains/rust-git-dev`, 1.94.0-dev |
| Preserved LLVM | `../rust/build/x86_64-unknown-linux-gnu/ci-llvm`, 21.1.8 |
| Backend base | `../ctng-alpha/rustc_codegen_gcc`, `02f889aec5dc9b4e51d4cfd86c67ce5e4a0bbf72` plus saved Alpha asm changes |
| Cross GCC/JIT | `/home/lindholm/x-tools/alphaev6-unknown-linux-gnu`, GCC 16.0.0 20251008, configured EV6 |
| libc fork base | 0.2.178, commit `2b95cbe1cecd28ab03d689b34c2696d70b65ff27` |
| Working Rust source/compiler | `../work/rust-src`, `../work/rust-build/x86_64-unknown-linux-gnu/stage1` |
| Working backend | `../work/backend-src`, `../work/backend-target/release/librustc_codegen_gcc.so` |

Portable diffs are in `patches/rust-alpha-abi-std.patch`,
`patches/libc-alpha.patch`, and `patches/rustc_codegen_gcc-symbol-versions.patch`.
The earlier inline-asm diff is preserved separately. The rebuild script supplies
matching rustc-dev metadata and host std files for the out-of-tree backend, and
installs the libgccjit symlink within the isolated compiler's sysroot.
The newer sibling `../rustc_codegen_gcc/` checkout was not used.

## Remaining limits

- Use `target_specs/alphaev6-unknown-linux-gnu-std.json` through the scripts. The
  original built-in target still has a different i64 alignment and has not been
  validated as a substitute. The scripts use the compiler's configured EV6 CPU;
  an explicit `cpu: ev6` hits the old backend's unsupported `-march=ev6` option.
- Panic unwinding, every libc API, static linking, native compiler bootstrap,
  older Alpha CPUs, and a broad third-party crate corpus remain unvalidated.
- The existing cross compiler/JIT is an input artifact. Recovering its complete
  build recipe and the earlier JIT state-reset workaround remains separate work.
- Applications using additional glibc APIs may require newer symbol versions;
  successfully running these probes on glibc 2.41 is not a blanket compatibility
  claim for every application linked against the 2.42 sysroot.

The prior no_std baseline failures and diagnostics remain under `../work/logs/`.
The original tracking discussion is
[rustc_codegen_gcc #742](https://github.com/rust-lang/rustc_codegen_gcc/issues/742).
