#!/usr/bin/env bash
# Build and run the corrected compiler/libc/std regression suite under QEMU.
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
source "$scripts/alpha-fixed-env.sh"
source "$scripts/alpha-env.sh"
"$ALPHA_CC" -no-pie "$ALPHA_REPO/libc_probe/constants.c" -o "$ALPHA_WORK/alpha-constants"
"$scripts/alpha-run.sh" "$ALPHA_WORK/alpha-constants" > "$ALPHA_WORK/logs/c-constants.txt"
for profile in debug release; do
    flags=()
    if [[ $profile == release ]]; then flags+=(--release); fi
    for crate in abi_probe alpha_test libc_probe; do
        "$scripts/alpha-build.sh" "$ALPHA_REPO/$crate/Cargo.toml" "${flags[@]}" \
            > "$ALPHA_WORK/logs/validated-$crate-$profile-build.log" 2>&1
    done
    "$scripts/alpha-std-probe.sh" "${flags[@]}" \
        > "$ALPHA_WORK/logs/validated-std-$profile-build.log" 2>&1
    for test in alpha_test abi_scalars abi_pair_arg abi_pair_ret abi_triple_arg abi_triple_callback abi_extended layouts runtime; do
        echo "$profile/$test"
        "$scripts/alpha-run.sh" "$ALPHA_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/$test"
    done
    "$scripts/alpha-run.sh" "$ALPHA_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/constants" \
        > "$ALPHA_WORK/logs/libc-constants-$profile.txt"
    python3 "$ALPHA_REPO/libc_probe/compare_constants.py" "$ALPHA_WORK/logs/c-constants.txt" \
        "$ALPHA_WORK/logs/libc-constants-$profile.txt"
    "$scripts/alpha-run.sh" "$ALPHA_STD_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/alpha_std_probe"
done
