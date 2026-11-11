Private experimental Rust/Alpha integration with Gentoo Portage

Goal
  Install the experimental native toolchain as a Portage-owned package, then
  build supported Rust applications using ordinary emerge commands on Alpha.
  This is a private proof of concept, not official Gentoo or Rust Alpha support.
  The overlay and supporting scripts contain LLM-generated work.

Prepared packages
  dev-lang/rust-alpha-bin-1.102.0_pre20261111
    Owns /opt/rust-alpha-experimental: native Rust, Cargo, GCC/libgccjit,
    binutils, std and patched support crates. The binaries come from the
    already validated native toolchain archive. Unused libtool .la files are
    omitted and the installed file checksum manifest is regenerated.
  sys-apps/ripgrep-15.2.0-r1: rg, without optional PCRE2
  sys-apps/fd-10.5.0-r1: fd, completions enabled, system allocator
  app-shells/zoxide-0.10.0-r1: zoxide and shell integration files

How Portage selects this compiler
  The three application ebuilds inherit the private alpha-cargo.eclass and
  have a BDEPEND on dev-lang/rust-alpha-bin. The eclass selects the packaged
  cargo-alpha launcher, explicitly targets alphaev6-unknown-linux-gnu, and
  uses the bundled Alpha libc/Rustix/Nix fixes. The native compiler runs on
  the ES40; no x86_64 compiler is required there.
  Cargo builds offline from the prepared, vendored source bundles. Temporary
  files, Cargo home and target files stay inside Portage's build directory.
  This first configuration fixes panic=abort, LTO off and release profile
  settings. Arbitrary system RUSTFLAGS are not applied. ALPHA_RUST_JOBS in
  make.conf controls parallelism, default 2.
  System Rust packages and global compiler selection are not modified.
  Unmodified main-tree Rust ebuilds do not automatically select this compiler.
  More packages need adapted ebuilds and testing; packages requiring LLVM,
  rustdoc, unwinding or special Rust components need further work.

On the ES40, after transferring/extracting the completed bundle
  Run as root from the extracted rust-alpha-portage directory:
    ./setup-es40.sh --check
    ./setup-es40.sh --install
    emerge --ask dev-lang/rust-alpha-bin::rust-alpha
    FEATURES="${FEATURES} test" emerge --ask \
      sys-apps/ripgrep::rust-alpha \
      sys-apps/fd::rust-alpha \
      app-shells/zoxide::rust-alpha

  setup-es40.sh --install only copies the overlay, narrowly scoped keyword
  settings and distfiles. It does not merge any packages. It checks bundle
  checksums and refuses to overwrite different existing configuration files.
  The toolchain ebuild unpacks a native binary compiler; it does not attempt
  to bootstrap Rust on the ES40. Application ebuilds compile from source.
  Their test phase runs functional application smoke tests, not full upstream
  suites. Subsequent ordinary emerge commands can rebuild these packages.
  If /etc/portage/repos.conf or package.accept_keywords is a single file,
  manually merge the matching file from config/ instead of using the installer.
  If the compiler was previously extracted manually into /opt, preserve/move
  that unowned directory aside before asking Portage to take ownership.

Optional prebuilt installation
  binpkgs/ contains Portage GPKG packages and a Packages index. Check the ES40
  CHOST against the index before installing them. This build uses
  alphaev67-unknown-linux-gnu packaging metadata and EV6-compatible code,
  against the existing ES40-derived glibc 2.41 sysroot.
  If compatible, use the absolute path of this bundle's binpkgs directory:
    PKGDIR="$PWD/binpkgs" emerge --ask --usepkgonly \
      dev-lang/rust-alpha-bin::rust-alpha \
      sys-apps/ripgrep::rust-alpha \
      sys-apps/fd::rust-alpha \
      app-shells/zoxide::rust-alpha
  These private packages are unsigned; SHA256SUMS and GPKG internal checksums
  are provided. Existing local signature policy may require additional setup.
  Do not bypass CHOST or runtime compatibility checks. The original bundle
  retains its original metadata. For this ES40, hardware-tested application
  payloads have separate packages in /root/rust-alpha-install-20261111/es40-binpkgs
  with alpha-unknown-linux-gnu CHOST metadata and unchanged image hashes.
  The adaptation script and original metadata are retained as provenance.
  The current overlay requires glibc >=2.39, based on a 73-file ELF symbol
  audit. Physical hardware testing uses the ES40 runtime reporting glibc 2.41.
  This does not constitute testing every supported glibc version.

Validation and limits
  See validation.json and evidence/ for completed checks.
  Host-side Portage tests use an isolated minimal Alpha test profile, a local
  Portage Python installation and QEMU binfmt. Host sandbox/userpriv features
  are disabled for this harness. This does not validate the ES40's actual
  profile, dependency graph or sandbox configuration. The subsequent ES40
  installation passed with its normal Portage features: all four packages
  are installed and owned by Portage. Native C, C++ exception handling, Cargo
  compilation, Rust threads/atomics/file I/O, and installed application smoke
  tests passed. Application source rebuilds under the ES40 sandbox have not
  yet been tested. See ES40-INSTALL.txt and evidence/es40/.
  Full native compiler self-bootstrap, panic unwinding and rustdoc remain
  outside the current toolchain's supported scope.

Files and reproducibility
  overlay/: ebuilds, shared eclass, package Manifest hashes and smoke tests
  distfiles/: pinned offline source bundles and original compiler archive
  binpkgs/: ready-made Portage binary packages, built through ebuild phases
  config/: repository, package keyword and exact-version unmask configuration
  source-provenance/: upstream URLs/tags/hashes, original and prepared locks
  Workstation build scripts are in the project portage directory. They depend
  on the existing project toolchains; they are not an empty-machine bootstrap.
  Native application builds use the shipped eclass. The final binary-install
  setup guard is refreshed in saved package environments. Host-only GPKG
  repacking uses Portage to normalize file ownership to root and recalculate
  package checksums; no root-owned host files are changed.

Reference Gentoo mechanisms
  https://devmanual.gentoo.org/eclass-reference/rust.eclass/index.html
  https://devmanual.gentoo.org/eclass-reference/cargo.eclass/index.html
  https://packages.gentoo.org/packages/sys-apps/ripgrep
  https://packages.gentoo.org/packages/sys-apps/fd
  https://packages.gentoo.org/packages/app-shells/zoxide
