#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
native="$w/stage/opt/rust-alpha-experimental"
rust="$w/rust-build/alphaev6-unknown-linux-gnu/stage1"
export TMPDIR="$w/tmp" QEMU_LD_PREFIX="$p/work/sdk/release/alpha-sdk/sysroot"
export PATH="$native/bin:$PATH" LD_LIBRARY_PATH="$rust/lib:$native/lib"
export GCC_EXEC_PREFIX="$native/lib/gcc/"
export COMPILER_PATH="$native/libexec/gcc/alphaev6-unknown-linux-gnu/17.0.0:$native/bin"
export LIBRARY_PATH="$native/lib:$native/lib/gcc/alphaev6-unknown-linux-gnu/17.0.0:$QEMU_LD_PREFIX/usr/lib"
export CG_GCCJIT_LIBRARY_PATH="$native/lib/libgccjit.so" RUSTC_BOOTSTRAP=1
export CARGO_HOME="$w/cargo-native-home" CARGO_TARGET_DIR="$w/native-cargo-target"
export RUSTUP_HOME="$p/work/upstream-20261111/rustup" RUSTC="$rust/bin/rustc"
export RUSTFLAGS="-Cpanic=abort -Clto=off -Clinker=$w/native-cc"
export ALPHA_CC="$native/bin/gcc" ALPHA_AR="$native/bin/ar" ALPHA_SYSROOT="$QEMU_LD_PREFIX"
ulimit -c 0
for source in "$w/tests/native-abi-probe/src/bin/"*.rs; do
 name=$(basename -- "$source" .rs)
 "$native/bin/cargo" rustc --release --offline --bin "$name" --manifest-path "$w/tests/native-abi-probe/Cargo.toml" -- -Clink-arg=-nostdlib -Clink-arg=-nostartfiles -Clink-arg=-lc
 "$CARGO_TARGET_DIR/release/$name"
done
