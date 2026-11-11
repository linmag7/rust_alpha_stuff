#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/alpha-env.sh"
status=0
for profile in debug release; do
    opts=()
    if [[ $profile == release ]]; then opts+=(--release); fi
    for crate in alpha_test abi_probe; do
        "$ALPHA_REPO/scripts/alpha-build.sh" "$ALPHA_REPO/$crate/Cargo.toml" "${opts[@]}" \
            > "$ALPHA_WORK/logs/$crate-build-$profile.log" 2>&1 || {
                cat "$ALPHA_WORK/logs/$crate-build-$profile.log"
                exit 1
            }
    done
    for test in alpha_test abi_scalars abi_pair_arg abi_pair_ret abi_triple_arg abi_triple_callback; do
        rc=0
        "$ALPHA_REPO/scripts/alpha-run.sh" "$CARGO_TARGET_DIR/alphaev6-unknown-linux-gnu/$profile/$test" \
            > "$ALPHA_WORK/logs/$test-$profile-qemu.log" 2>&1 || rc=$?
        printf '%s/%s: exit %s\n' "$profile" "$test" "$rc"
        if (( rc != 0 )); then
            cat "$ALPHA_WORK/logs/$test-$profile-qemu.log"
            status=1
        fi
    done
done
exit "$status"
