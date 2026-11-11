#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$here/../../../.." && pwd)
w="$p/work/ecosystem-20261111"
n="$p/work/native-20261111"
mkdir -p "$w/logs" "$w/tmp"
export TMPDIR="$w/tmp"
export RUSTFLAGS="-Clinker=$n/alpha-link --cfg=rustix_use_libc --cfg=rustix_no_linux_raw"
ulimit -c 0
exec "$p/work/experimental-20261111/cargo-alpha" build \
 --manifest-path "$here/Cargo.toml" --target-dir "$w/cross-target" \
 --config "patch.crates-io.libc.path=\"$n/src/libc\"" \
 --config "patch.crates-io.rustix.path=\"$n/src/rustix-compiler\"" \
 --config "patch.crates-io.rustix_114.package=\"rustix\"" \
 --config "patch.crates-io.rustix_114.path=\"$n/src/rustix\"" \
 --config "patch.crates-io.nix_030.package=\"nix\"" \
 --config "patch.crates-io.nix_030.path=\"$n/src/nix-0.30.1\"" \
 --config "patch.crates-io.nix_031.package=\"nix\"" \
 --config "patch.crates-io.nix_031.path=\"$n/src/nix-0.31.3\"" \
 --config "patch.crates-io.openssl-src.path=\"$n/src/openssl-src\"" \
 -j8 "$@"
