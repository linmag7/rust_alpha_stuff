#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
native="$w/stage/opt/rust-alpha-experimental"
export TMPDIR="$w/tmp" QEMU_LD_PREFIX="$p/work/sdk/release/alpha-sdk/sysroot" ALPHA_SYSROOT="$p/work/sdk/release/alpha-sdk/sysroot"
export CARGO_HOME="$w/package-cargo-home" CARGO_TARGET_DIR="$w/package-cargo-target" RUSTUP_HOME="$p/work/upstream-20261111/rustup"
export PATH="$native/bin:$PATH" LD_LIBRARY_PATH="$native/lib"
ulimit -c 0
"$native/bin/rustc" --version --verbose
"$native/bin/rustc" --print sysroot
"$native/bin/cargo-alpha" --version
"$native/bin/cargo-alpha" run --offline --manifest-path "$w/tests/packaged-probe/Cargo.toml"
"$native/bin/cargo-alpha" run --offline --manifest-path "$w/tests/native-macro-app/Cargo.toml"
"$native/bin/cargo-alpha" test --offline --manifest-path "$w/tests/native-macro-app/Cargo.toml"
echo 'Staged native toolchain, dependency patches, proc macro and unit test: PASS'
