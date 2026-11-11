#!/usr/bin/env bash
# Source this file from scripts; existing installed tools are read-only inputs.
ALPHA_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ALPHA_PROJECT=${ALPHA_PROJECT:-$(dirname -- "$ALPHA_REPO")}
ALPHA_WORK=${ALPHA_WORK:-$ALPHA_PROJECT/work}
ALPHA_TOOLCHAIN=${ALPHA_TOOLCHAIN:-/home/lindholm/toolchains/rust-git-dev}
ALPHA_CROSS=${ALPHA_CROSS:-/home/lindholm/x-tools/alphaev6-unknown-linux-gnu}
ALPHA_BACKEND=${ALPHA_BACKEND:-$ALPHA_PROJECT/ctng-alpha/rustc_codegen_gcc/target/release/librustc_codegen_gcc.so}
ALPHA_SYSROOT=${ALPHA_SYSROOT:-$ALPHA_CROSS/alphaev6-unknown-linux-gnu/sysroot}
ALPHA_TARGET=${ALPHA_TARGET:-$ALPHA_REPO/target_specs/alphaev6-unknown-linux-gnu.json}
export TMPDIR=$ALPHA_WORK/tmp
export CARGO_HOME=$ALPHA_WORK/cargo-home
export CARGO_TARGET_DIR=${ALPHA_TARGET_DIR:-$ALPHA_WORK/target}
export RUSTC=$ALPHA_TOOLCHAIN/bin/rustc
export LD_LIBRARY_PATH=$ALPHA_TOOLCHAIN/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$ALPHA_CROSS/bin:$PATH
export ALPHA_CC=$ALPHA_CROSS/bin/alphaev6-unknown-linux-gnu-gcc
export ALPHA_AR=$ALPHA_CROSS/bin/alphaev6-unknown-linux-gnu-ar
export ALPHA_SYSROOT
mkdir -p "$TMPDIR" "$CARGO_HOME" "$CARGO_TARGET_DIR" "$ALPHA_WORK/logs"
# Cargo's encoded form preserves paths containing spaces.
alpha_flags=("-Zcodegen-backend=$ALPHA_BACKEND" "--sysroot=$ALPHA_TOOLCHAIN"
    '-C' "linker=$ALPHA_CC" '-C' "link-arg=--sysroot=$ALPHA_SYSROOT"
    '-C' 'link-arg=-nostdlib' '-C' 'link-arg=-nostartfiles' '-C' 'link-arg=-lc')
printf -v CARGO_ENCODED_RUSTFLAGS '%s\x1f' "${alpha_flags[@]}"
export CARGO_ENCODED_RUSTFLAGS=${CARGO_ENCODED_RUSTFLAGS%$'\x1f'}
unset RUSTFLAGS
