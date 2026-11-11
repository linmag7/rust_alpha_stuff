#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
export CXX_alphaev6_unknown_linux_gnu="$w/cross/bin/alphaev6-unknown-linux-gnu-g++"
export LIBZ_SYS_STATIC=1
export RUSTFLAGS="-Clinker=$w/alpha-link --cfg=rustix_use_libc --cfg=rustix_no_linux_raw"
exec "$p/work/experimental-20261111/cargo-alpha" build \
 --manifest-path "$w/src/rust/src/tools/cargo/Cargo.toml" \
 --config "patch.crates-io.openssl-src.path=\"$w/src/openssl-src\"" \
 --config "patch.crates-io.rustix.path=\"$w/src/rustix\"" \
 --config "patch.crates-io.libc.path=\"$w/src/libc\"" \
 --target-dir "$w/cargo-target" --release --bin cargo --features all-static -j8 "$@"
