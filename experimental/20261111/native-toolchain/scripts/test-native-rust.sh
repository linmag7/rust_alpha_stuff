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
"$rust/bin/rustc" --version --verbose
"$rust/bin/rustc" "$w/tests/native-rust-std.rs" -Cpanic=abort -Clinker="$w/native-cc" -o "$w/tests/native-rust-std"
"$w/tests/native-rust-std"
export CARGO_HOME="$w/cargo-native-home" CARGO_TARGET_DIR="$w/native-cargo-target"
export RUSTUP_HOME="$p/work/upstream-20261111/rustup" RUSTC="$rust/bin/rustc"
export RUSTFLAGS="-Cpanic=abort -Clto=off -Clinker=$w/native-cc"
"$native/bin/cargo" run --offline --manifest-path "$w/tests/native-macro-app/Cargo.toml"
