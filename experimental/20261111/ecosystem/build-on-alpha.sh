#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
# Override this when the experimental toolchain is installed elsewhere.
export PATH="${ALPHA_TOOLCHAIN_PREFIX:-/opt/rust-alpha-experimental}/bin:$PATH"
export CARGO_HOME="${CARGO_HOME:-$PWD/.cargo-home}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/target}"
export TMPDIR="${TMPDIR:-$PWD/tmp}"
mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR" "$TMPDIR"
ulimit -c 0
cargo-alpha build --locked --offline -j"${JOBS:-2}" "$@"
profile=debug
for arg in "$@"; do
 if [[ "$arg" == --release ]]; then profile=release; fi
done
for probe in foundations concurrency network database; do
 "$CARGO_TARGET_DIR/$profile/$probe"
done
