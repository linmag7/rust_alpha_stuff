#!/usr/bin/env bash
# Source this before building with the isolated ABI/std toolchain.
_alpha_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export ALPHA_PROJECT=${ALPHA_PROJECT:-$(dirname -- "$_alpha_repo")}
export ALPHA_WORK=${ALPHA_WORK:-$ALPHA_PROJECT/work}
export ALPHA_TOOLCHAIN=${ALPHA_TOOLCHAIN:-$ALPHA_WORK/rust-build/x86_64-unknown-linux-gnu/stage1}
export ALPHA_BACKEND=${ALPHA_BACKEND:-$ALPHA_WORK/backend-target/release/librustc_codegen_gcc.so}
_alpha_id=current
if [[ -f $ALPHA_WORK/toolchain-id ]]; then read -r _alpha_id < "$ALPHA_WORK/toolchain-id"; fi
# Refresh automatically chosen paths when this file is sourced again after a rebuild.
if [[ ${ALPHA_TOOLCHAIN_CACHE_ID:-$_alpha_id} != $_alpha_id ]]; then
    if [[ ${ALPHA_TARGET_DIR:-} == "$ALPHA_WORK/validated-$ALPHA_TOOLCHAIN_CACHE_ID-target" ]]; then unset ALPHA_TARGET_DIR; fi
    if [[ ${ALPHA_STD_TARGET_DIR:-} == "$ALPHA_WORK/validated-$ALPHA_TOOLCHAIN_CACHE_ID-std-target" ]]; then unset ALPHA_STD_TARGET_DIR; fi
fi
export ALPHA_TOOLCHAIN_CACHE_ID=$_alpha_id
export ALPHA_TARGET_DIR=${ALPHA_TARGET_DIR:-$ALPHA_WORK/validated-$_alpha_id-target}
export ALPHA_STD_TARGET_DIR=${ALPHA_STD_TARGET_DIR:-$ALPHA_WORK/validated-$_alpha_id-std-target}
unset _alpha_id
mkdir -p "$ALPHA_WORK"
cp "$_alpha_repo/target_specs/alphaev6-unknown-linux-gnu-std.json" "$ALPHA_WORK/alphaev6-unknown-linux-gnu.json"
export ALPHA_TARGET=$ALPHA_WORK/alphaev6-unknown-linux-gnu.json
unset _alpha_repo
