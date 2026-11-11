# Experimental Rust on Linux/Alpha

This snapshot builds Rust with `rustc_codegen_gcc` and GCC/libgccjit, both as an
x86_64-hosted cross-toolchain and as a native Alpha toolchain. The native compiler,
Cargo, standard library, GCC and binutils have been installed on an AlphaServer
ES40 running Gentoo. This is a personal proof of concept, not supported Rust or
Gentoo Alpha support and not an upstream submission. It contains LLM-generated
implementation and supporting material.

## Current results

- Rust 1.102.0-dev, Cargo 1.101.0, GCC 17 development and upstream binutils
  2.47.50.20261111; target `alphaev6-unknown-linux-gnu` (EV6 or newer).
- Alpha C ABI classification, inline assembly, target layout and libc bindings
  support `std`; additional changes cover metadata, bootstrap and support crates.
- Native C, C++ exception handling and Cargo/Rust compilation passed on the
  ES40. The Rust program exercised threads, atomics and UTF-8 file I/O.
- ripgrep 15.2.0, fd 10.5.0 and zoxide 0.10.0 are installed through Portage and
  passed functional checks on the actual machine.
- Native application source builds through Portage passed under QEMU. The
  physical ES40 zoxide source rebuild is in progress; that result is not yet
  claimed as a pass.

See [installation results](experimental/20261111/portage/ES40-INSTALL.txt),
[test evidence](experimental/20261111/portage/evidence/es40), and
[the native toolchain record](experimental/20261111/native-toolchain/status.json).
Earlier cross-build results are in [the historical overview](HISTORICAL-CROSS-TOOLCHAIN.md).

## Sources

The publication branch is `alpha-experimental-20261111` in each repository:

| Repository | Purpose |
| --- | --- |
| [rust_alpha_stuff](https://github.com/linmag7/rust_alpha_stuff) | Patches, probes, provenance, build scripts and Gentoo overlay |
| [rust](https://github.com/linmag7/rust) | Native Rust target/std/bootstrap snapshot, including its in-tree GCC backend |
| [rustc_codegen_gcc](https://github.com/linmag7/rustc_codegen_gcc) | Standalone backend used for the cross-build checkpoint |
| [gcc](https://github.com/linmag7/gcc) | Alpha JIT assembler relocation-sequence fix and regression test |

[Exact source revisions](publication/source-revisions.json) distinguish upstream
bases from the experimental commits. The Rust fork includes its own in-tree
backend; the standalone backend branch is not interchangeable with that subtree.
There are no local binutils source changes in this snapshot. Existing upstream
Alpha fixes are used rather than copied into a local binutils patch set.

The libc and other dependency changes are supplied as patches in
[the baseline patch set](experimental/20261111/patches) and
[the native additions](experimental/20261111/native-toolchain/patches).
Cargo's prepared lockfile is in `experimental/20261111/native-toolchain/locks`.

## Trying the native toolchain

Read [the Gentoo installation guide](GENTOO-SETUP.txt) first. It explains the
required archives, checksums, overlay registration and source-build command.
The repository contains source and configuration, not compiler binaries.
The installer needs four prepared distfiles; do not assume a Git clone alone
is sufficient. The accompanying install kit is prepared separately for release.

With the native toolchain installed, use:

```sh
/opt/rust-alpha-experimental/bin/cargo-alpha new --vcs none hello-alpha
cd hello-alpha
/opt/rust-alpha-experimental/bin/cargo-alpha run
```

The wrapper selects GCC, libgccjit and the patched support crates. It does not
replace the system compiler. Unused-patch warnings are expected when a project
does not depend on those support crates.

For rebuilding the supported Gentoo package, as root:

```sh
FEATURES="${FEATURES} test" emerge --ask --oneshot --usepkg=n app-shells/zoxide::rust-alpha
```

## Portage integration boundaries

`dev-lang/rust-alpha-bin` owns `/opt/rust-alpha-experimental`. The adapted
application ebuilds inherit `alpha-cargo.eclass`, depend on that compiler, and
select its `cargo-alpha` wrapper. Vendored dependencies allow offline builds.
Per-package keywords and exact-version unmask entries enable only our packages.

This is **not** a `virtual/rust` provider and is not registered with
`eselect rust` or Gentoo's general slotted `rust.eclass`. Arbitrary unmodified
Rust ebuilds do not automatically work. Full integration needs compiler-slot
packaging and selection, component/feature compatibility, and more package tests.
The current package does not provide rustdoc, clippy or rustfmt.

## Limits and reproduction

Rust panics abort; Rust unwinding and doctests are unsupported. Executables use
non-PIE defaults while the loader/linker issue is investigated. LTO and
incremental compilation are disabled in the tested configuration. Rust C-variadic
function definitions and reliable f128 are unavailable. Passing these probes is
not a full compiler correctness claim or a native self-bootstrap.

[Build scripts and configurations](experimental/20261111/native-toolchain/scripts)
record the actual builds. They expect the prepared workspace and Alpha sysroot;
archived TOML files contain the original absolute paths. They are not yet an
empty-machine bootstrap installer. The source snapshots and patches support
inspection and reproducing the experiment with those prerequisites.

For upstream discussions, authors should write their own account of the work
and follow each project's contribution policies. These experimental branches
are not requests for upstream maintainers to review or accept generated code.
