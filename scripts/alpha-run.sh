#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/alpha-env.sh"
binary=${1:-$CARGO_TARGET_DIR/alphaev6-unknown-linux-gnu/debug/alpha_test}
if (( $# )); then shift; fi
ulimit -c 0
exec timeout "${ALPHA_TIMEOUT:-30}" "${ALPHA_QEMU:-qemu-alpha-static}" -L "$ALPHA_SYSROOT" "$binary" "$@"
