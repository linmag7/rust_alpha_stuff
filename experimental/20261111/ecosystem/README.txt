Common Rust crates on experimental Alpha Linux

This private proof of concept builds 22 direct dependencies into four runtime
probes. Cargo.lock pins the complete dependency graph. It uses the existing
experimental Rust/GCC backend and Alpha libc/Rustix patches, panic=abort and
non-PIE executables. The source and supporting text include LLM-generated work.

Unlike std, these libraries are ordinary Cargo dependencies. Their compiled
artifacts depend on the compiler, profile and enabled features. Do not copy
rlib files into the compiler sysroot. Reuse Cargo.toml/Cargo.lock and build
with cargo-alpha; Cargo manages the resulting build cache.

Validation completed
  All four probes passed under QEMU EV6 in cross-built debug, cross-built
  release and native-built debug configurations: 12 successful program runs.
  Native Cargo/rustc/GCC compiled the full selected dependency graph, including
  procedural macros and bundled SQLite C source, under QEMU. No new crate
  patches were needed beyond the existing experimental libc/Rustix patches.
  Offline native Cargo resolution passed with an empty Cargo cache. All 10,568
  vendored files in 195 registry crates passed checksum verification.
  See CRATES.txt for exact direct versions and features; validation.json
  records runtime checks and compiler provenance.

Coverage
  foundations: serde + serde_json + toml round trips, clap derive and invalid
    input, Unicode regex, anyhow/thiserror errors, tracing/tracing-subscriber,
    SHA-256 known vector, rand seeded/OS randomness, UUID v4, gzip round trip,
    tempfile creation/cleanup and walkdir traversal.
  concurrency: Rayon parallel reduction, crossbeam-channel bounded messages,
    parking_lot contended Mutex and RwLock.
  network: Tokio multithread tasks/timers/channels/blocking pool/async IO,
    Axum serving Reqwest HTTP+JSON requests over loopback TCP, HTTP 404,
    graceful server shutdown and UDP.
  database: rusqlite with bundled SQLite C source, WAL, commit/rollback,
    close/reopen, integer/text/blob/float values and integrity_check.

Limits
  These are integration smoke tests, not the upstream crates' full suites.
  Only the versions/features in Cargo.toml and Cargo.lock are covered.
  Reqwest has default features disabled: plain HTTP is tested, TLS is not.
  No ES40 hardware validation or native Rust compiler self-bootstrap is claimed.
  Existing experimental toolchain limitations still apply.

Project workspace commands (from this directory)
  ./build-cross.sh --locked --offline
  ./run-qemu.py
  ./build-cross.sh --release --locked --offline
  ./run-qemu.py --profile release
  ./build-native.sh
  ./run-qemu.py --mode native

The cross/native build scripts use the prepared toolchains in this project.
Work, caches and logs stay under the project disk. Results are recorded under
work/ecosystem-20261111; validation.json contains the exported evidence.

Offline source bundle / later ES40 use
  prepare-offline.py copies this example and vendors registry dependencies.
  The separate native Alpha toolchain supplies the patched libc/Rustix sources.
  Extract alpha-ecosystem-offline-20261111.tar.xz, then:
    cd alpha-ecosystem
    ./build-on-alpha.sh
  Default toolchain prefix: /opt/rust-alpha-experimental. Override with
  ALPHA_TOOLCHAIN_PREFIX if needed. JOBS defaults to 2 for the ES40; adjust for
  its available memory. Build caches stay inside the extracted directory unless
  CARGO_HOME/CARGO_TARGET_DIR/TMPDIR are explicitly set. The network probe only
  contacts local loopback and does not need an Internet connection.
  A configured native Alpha libc development environment remains necessary.

Selection references
  https://rust-cli.github.io/book/resources/index.html
  https://docs.rs/tokio/latest/tokio/

The final offline archive also includes prebuilt/cross-release and
prebuilt/native-debug probes. These require a compatible Alpha runtime
(the current tests use glibc 2.41). They can be run individually to check
hardware before spending time rebuilding. Source build dependencies still
require the separate experimental native toolchain and libc development files.
