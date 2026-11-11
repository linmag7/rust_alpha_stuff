#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
stage="$w/host-rust-build/x86_64-unknown-linux-gnu/stage1"
export TMPDIR="$w/tmp" CARGO_HOME="$w/cargo-home" CARGO_TARGET_DIR="$w/host-backend-target"
export RUSTC="$stage/bin/rustc" RUSTC_BOOTSTRAP=1
export LD_LIBRARY_PATH="$stage/lib" LIBRARY_PATH="$stage/lib"
"$w/host-rust-build/x86_64-unknown-linux-gnu/stage0/bin/cargo" build --release \
 --manifest-path "$w/src/rust/compiler/rustc_codegen_gcc/Cargo.toml" -j12 "$@"
