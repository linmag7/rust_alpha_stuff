#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
export TMPDIR="$w/tmp" CARGO_HOME="$w/cargo-home"
export RUSTUP_HOME="$p/work/upstream-20261111/rustup"
cd "$w/src/rust"
./x build --config "$w/host-rust-config.toml" --stage 2 compiler/rustc library/std --jobs 24 "$@"
