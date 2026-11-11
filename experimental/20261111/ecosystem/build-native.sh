#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$here/../../../.." && pwd)
w="$p/work/ecosystem-20261111"
n="$p/work/native-20261111/stage/opt/rust-alpha-experimental"
mkdir -p "$w/logs" "$w/tmp"
export TMPDIR="$w/tmp" QEMU_LD_PREFIX="$p/work/sdk/release/alpha-sdk/sysroot" ALPHA_SYSROOT="$p/work/sdk/release/alpha-sdk/sysroot"
export CARGO_HOME="$p/work/native-20261111/package-cargo-home" CARGO_TARGET_DIR="$w/native-target"
export RUSTUP_HOME="$p/work/upstream-20261111/rustup"
ulimit -c 0
exec "$n/bin/cargo-alpha" build --manifest-path "$here/Cargo.toml" --locked --offline -j4 "$@"
