#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
source "$scripts/alpha-fixed-env.sh"
source "$scripts/alpha-env.sh"
bundle=$ALPHA_WORK/es40-abi-std
mkdir -p "$bundle"/{debug,release}
for profile in debug release; do
    for test in alpha_test abi_scalars abi_pair_arg abi_pair_ret abi_triple_arg abi_triple_callback abi_extended layouts runtime constants; do
        cp "$ALPHA_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/$test" "$bundle/$profile/"
    done
    cp "$ALPHA_STD_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/alpha_std_probe" "$bundle/$profile/"
done
cp "$scripts/run-fixed-tests-native.sh" "$bundle/run-tests-native.sh"
cp "$ALPHA_REPO/libc_probe/"{compare_constants.py,unavailable-constants.json,constants.c,layouts.c,layouts-dump.c} "$bundle/"
"$ALPHA_CC" -O2 -no-pie "$bundle/constants.c" -o "$bundle/c-constants"
"$ALPHA_CC" -O2 -no-pie "$bundle/layouts-dump.c" -o "$bundle/c-layouts"
(cd "$bundle" && sha256sum debug/* release/* c-constants c-layouts > SHA256SUMS)
tar -czf "$ALPHA_WORK/es40-abi-std.tar.gz" -C "$bundle" .
echo "$ALPHA_WORK/es40-abi-std.tar.gz"
