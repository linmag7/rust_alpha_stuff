#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
export RUSTFLAGS="-Clinker=$w/alpha-link --cfg=rustix_use_libc --cfg=rustix_no_linux_raw"
"$p/work/experimental-20261111/cargo-alpha" build --release \
 --manifest-path "$w/tests/rustix-probe/Cargo.toml" --target-dir "$w/cargo-target" \
 --config "patch.crates-io.libc.path=\"$w/src/libc\"" \
 --config "patch.crates-io.rustix.path=\"$w/src/rustix\""
"$p/work/binutils-20261111/alpha-cc" "$w/tests/ioctl-constants.c" -no-pie -o "$w/tests/ioctl-constants"
export QEMU_LD_PREFIX="$p/work/sdk/release/alpha-sdk/sysroot"
"$w/tests/ioctl-constants" > "$w/logs/ioctl-c.txt"
"$w/cargo-target/alphaev6-unknown-linux-gnu/release/native-alpha-rustix-probe" > "$w/logs/ioctl-rust.txt"
diff -u "$w/logs/ioctl-c.txt" "$w/logs/ioctl-rust.txt"
echo 'Alpha ioctl C/Rust comparison, SYS_open Rustix open/fstat and Nix signal checks: PASS'
