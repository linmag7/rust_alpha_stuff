#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
export TMPDIR="$w/tmp" CARGO_HOME="$w/cargo-home"
export RUSTUP_HOME="$p/work/upstream-20261111/rustup"
export CG_GCCJIT_LIBRARY_PATH="$p/work/rebase-20261111/build/gcc/gcc/libgccjit.so"
export LD_LIBRARY_PATH="$w/host-rust-build/x86_64-unknown-linux-gnu/stage2/lib"
export CARGO_INCREMENTAL=0
cd "$w/src/rust"
./x build --config "$w/rust-config.toml" --stage 1 compiler/rustc --jobs 12 "$@"
python3 "$w/install-native-std.py"
