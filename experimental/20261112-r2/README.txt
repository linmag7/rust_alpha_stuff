Experimental Rust on Alpha — maintenance kit, 2026-11-12 revision 2

This is a personal proof of concept, not official Rust/Gentoo Alpha support.
The kit contains LLM-assisted code and documentation. Review before reuse.
Native host/target: Alpha EV6 or newer Linux GNU; tested on a Gentoo ES40/EV67,
glibc2.41. ELF audit of the original compiler required glibc>=2.39; other runtimes
have not been tested. Native C headers/libraries and working Gentoo GCC/Portage,
Python3, bash, tar/xz and network access to Gentoo distfiles are required.

What is in this update
  Native Rust1.102.0-dev, Cargo1.101.0, GCC17/libgccjit, upstream binutils
  2.47.50.20261111 and Alpha std. The GCC backend is rebuilt with the compare_bytes size_t fix (7cb1ce3).
  The compiler driver/std and other toolchain binaries retain the November11 build.
  cargo-alpha now defaults to IEEE arithmetic (-mieee). alpha-cc/alpha-c++
  use five bundled Gentoo GCC15 startup objects to fix __dso_handle linking.
  Startup object provenance and hashes are included; this is a temporary
  workaround pending a GCC rebuild with --enable-__cxa_atexit.
  Shared alpha-rust.eclass, versioned crate patches, dependency preflight,
  tested application ebuilds, exact keyword/unmask/USE configuration, checksums.
  Original compiler/source archives are retained for provenance and rollback.
  Recent application crate archives are fetched by Portage with Manifest hashes;
  the whole application collection is NOT an offline mirror or binary repository.

Install on Alpha Gentoo, as root
  tar -xf rust-alpha-native-portage-20261112-r2.tar
  cd rust-alpha-portage-20261112-r2
  ./setup-es40.sh --check
  ./setup-es40.sh --install
  # Existing rust-alpha overlay: review changes, then use --update instead.
  # --update backs up the old overlay and /etc/portage before copying updates.
  emerge --ask =dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2::rust-alpha
  emerge --ask sys-apps/bat::rust-alpha sys-block/dust::rust-alpha

The installer verifies SHA256SUMS and installs overlay/config/distfiles only.
It does not merge packages or replace system tools. Inspect every emerge plan;
do not accept unrelated upgrades blindly. Existing single-file Portage config
layouts require manual merging of the supplied entries. Compiler install lives
under /opt/rust-alpha-experimental and is owned by rust-alpha-bin. If that prefix
was manually extracted and is unowned, move it aside before the first emerge.
The old compiler revision stays available for rollback. There are no newly
repacked application binpkgs in this release. No public release download has
been uploaded yet; obtain the kit from the experiment owner.

Compile your own Rust program natively
  /opt/rust-alpha-experimental/bin/cargo-alpha new --vcs none hello-alpha
  cd hello-alpha
  /opt/rust-alpha-experimental/bin/cargo-alpha run --release

Use cargo-alpha rather than plain cargo to select the GCC backend environment,
patched libc/Rustix/Nix sources, native linker and supported flags. Existing
Cargo.lock files may need updating to the compatible patched crate versions.
The eclass's additional private sysinfo/terminal_size/nix patches are opt-in;
they are NOT automatically injected into every arbitrary Cargo project.
See RUST-BUILD-RECIPE.txt and overlay/eclass/alpha-rust for adapting new packages.
Rust USE flags or unmasking alone do not adapt a main-tree ebuild to this SDK.

Supported application recipes (Alpha builds and ES40 smoke checks)
  sys-apps/ripgrep15.2.0-r1 (without PCRE2), sys-apps/fd10.5.0-r1,
  app-shells/zoxide0.10.0-r1, app-shells/fish4.9.3-r2,
  app-editors/helix25.07.1-r1 (grammars), sys-process/bottom0.14.6-r1
  (core monitor, optional GPU/battery/ZFS disabled), dev-util/git-delta0.19.2-r1,
  app-benchmarks/hyperfine1.21.0-r1, sys-block/dust1.2.3-r1, sys-apps/bat0.26.1-r2.
  Also dev-vcs/git2.56.1-r1 with Rust enabled; Git is not a Rust-only application.
  Initial ripgrep/fd/zoxide packaging used QEMU; recent Fish-through-Bat
  builds ran natively through emerge. See packages.json and VALIDATION.txt. Full upstream test suites were not run.
  No login-shell changes are made. Test Fish interactively before adopting it.

Build/repack provenance
  The source forks use branch alpha-experimental-20261111 in linmag7/rust,
  linmag7/gcc, linmag7/rustc_codegen_gcc and linmag7/rust_alpha_stuff.
  Earlier compiler build scripts, locked revisions and patches are under
  experimental/20261111/native-toolchain/{scripts,locks,patches} and the
  cross compiler work under experimental/20261111. Read those README files
  as historical snapshots; later hardware results are recorded in this release.
  Those compiler build scripts expect the prepared gcc-alpha-rust workspace,
  sysroot and bootstrap tools: they are not a clean-machine bootstrap recipe.
  For this maintenance repack: extract the original native compiler into STAGE,
  replace bin/{cargo-alpha,alpha-cc,alpha-c++} from wrappers/, add the supplied
  inputs/crt.tar under share/rust-alpha-experimental/support, copy the eclass
  compatibility patch bank there, then run:
    python3 repack-sdk.py STAGE OUTPUT.tar.xz
  The full release assembly recipe and input hashes are in release-inputs.json
  and scripts/. Repacking builds no compiler. Prefer Portage for installation.

Limits
  panic=abort; LTO/incremental off in supported recipes; rustdoc/doctests and
  native compiler self-bootstrap unsupported/unvalidated. Raw syscall crates
  need separate Alpha ABI work; use libc-backed code. Native executables default
  non-PIE due to an unresolved small-PIE loader/linker issue. Full std was not
  rebuilt with the new IEEE defaults. Existing GCC memory-access warnings and
  broader C ABI/FFI correctness still need investigation. Passing application
  smoke checks is not full compiler correctness or upstream-readiness evidence.
