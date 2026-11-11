#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/alpha-env.sh"
manifest=${1:-$ALPHA_REPO/alpha_test/Cargo.toml}
if (( $# )); then shift; fi
exec "$ALPHA_TOOLCHAIN/bin/cargo" build --offline --manifest-path "$manifest" \
    -Zbuild-std=core,alloc,compiler_builtins \
    -Zbuild-std-features=compiler-builtins-no-f16-f128 \
    --target "$ALPHA_TARGET" "$@"
