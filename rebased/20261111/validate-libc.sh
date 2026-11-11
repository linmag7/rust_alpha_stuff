#!/usr/bin/env bash
# Validate rebased libc with the previously validated Rust/GCC toolchain.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
project=$(dirname -- "$repo")
rebase=$project/work/rebase-20261111
export ALPHA_TOOLCHAIN=$rebase/install/validated-rust
export ALPHA_CROSS=/home/lindholm/x-tools/alphaev6-unknown-linux-gnu
export ALPHA_SYSROOT=/home/lindholm/x-tools/alphaev6-unknown-linux-gnu/alphaev6-unknown-linux-gnu/sysroot
export ALPHA_TARGET_DIR=$rebase/build/libc-isolated-validation
export ALPHA_STD_TARGET_DIR=$rebase/build/libc-isolated-std-validation
source "$repo/scripts/alpha-fixed-env.sh"
source "$repo/scripts/alpha-env.sh"
"$ALPHA_CC" -no-pie "$rebase/probes/libc_probe/constants.c" -o "$rebase/build/c-constants"
"$repo/scripts/alpha-run.sh" "$rebase/build/c-constants" > "$rebase/logs/c-constants.txt"
for profile in debug release; do
    flags=()
    if [[ $profile == release ]]; then flags+=(--release); fi
    for crate in libc_probe abi_probe alpha_test; do
        "$repo/scripts/alpha-build.sh" "$rebase/probes/$crate/Cargo.toml" "${flags[@]}" \
            > "$rebase/logs/$crate-libc-$profile-build.log" 2>&1
    done
    for test in layouts runtime alpha_test abi_scalars abi_pair_arg abi_pair_ret abi_triple_arg abi_triple_callback abi_extended; do
        echo "$profile/$test"
        "$repo/scripts/alpha-run.sh" "$ALPHA_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/$test"
    done
    "$repo/scripts/alpha-run.sh" "$ALPHA_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/constants" \
        > "$rebase/logs/libc-constants-$profile.txt"
    python3 "$rebase/probes/libc_probe/compare_constants.py" "$rebase/logs/c-constants.txt" \
        "$rebase/logs/libc-constants-$profile.txt"
    "$repo/scripts/alpha-std-build.sh" "$rebase/probes/std_probe/Cargo.toml" "${flags[@]}" \
        --config "patch.crates-io.libc.path=\"$rebase/src/libc\"" \
        > "$rebase/logs/libc-std-$profile-build.log" 2>&1
    "$repo/scripts/alpha-run.sh" "$ALPHA_STD_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/alpha_std_probe"
done
