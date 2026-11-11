#!/usr/bin/env bash
# Cross-build a Cargo application with the patched Alpha std and libc.
set -euo pipefail
scripts=$(cd -- "$(dirname -- "$0")" && pwd)
source "$scripts/alpha-fixed-env.sh"
source "$scripts/alpha-env.sh"
manifest=${1:-$ALPHA_REPO/std_probe/Cargo.toml}
if (( $# )); then shift; fi
flags=("-Zcodegen-backend=$ALPHA_BACKEND" "--sysroot=$ALPHA_TOOLCHAIN"
    '-C' "linker=$ALPHA_CC" '-C' "link-arg=--sysroot=$ALPHA_SYSROOT")
printf -v CARGO_ENCODED_RUSTFLAGS '%s\x1f' "${flags[@]}"
export CARGO_ENCODED_RUSTFLAGS=${CARGO_ENCODED_RUSTFLAGS%$'\x1f'}
export CARGO_TARGET_DIR=$ALPHA_STD_TARGET_DIR
exec "$ALPHA_TOOLCHAIN/bin/cargo" build --offline --manifest-path "$manifest" \
    -Zbuild-std=std,panic_abort \
    -Zbuild-std-features=compiler-builtins-no-f16-f128 \
    --target "$ALPHA_TARGET" \
    --config "patch.crates-io.libc.path=\"$ALPHA_REPO/libc-alpha\"" "$@"
