#!/usr/bin/env bash
# Build the SDK-only path-discovery adjustment without modifying the baseline.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$here/../.." && pwd)
work=$project/work/sdk
export TMPDIR=$work/tmp CARGO_HOME=$project/work/cargo-home
export CARGO_TARGET_DIR=$work/backend-target
export RUSTC=$project/work/rust-build/x86_64-unknown-linux-gnu/stage1/bin/rustc
export RUSTC_BOOTSTRAP=1 LD_LIBRARY_PATH=$project/work/rust-build/x86_64-unknown-linux-gnu/stage1/lib
mkdir -p "$TMPDIR" "$work/logs"
if [[ ! -d $work/backend-src ]]; then
    mkdir -p "$work/backend-src"
    rsync -a --exclude=.git --exclude=target "$project/work/backend-src/" "$work/backend-src/"
    patch -d "$work/backend-src" -p1 < "$here/backend-relocatable-lto.patch"
fi
"$project/work/rust-build/x86_64-unknown-linux-gnu/stage1/bin/cargo" build --offline --release \
    --manifest-path "$work/backend-src/Cargo.toml"
