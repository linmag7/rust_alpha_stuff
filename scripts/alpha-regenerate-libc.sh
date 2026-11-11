#!/usr/bin/env bash
# Development only: regenerate the checked-in bindings/probes from Alpha headers.
# Requires pyelftools 0.32; normal builds and validation do not require it.
set -euo pipefail
source "$(dirname -- "$0")/alpha-env.sh"
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$ALPHA_WORK/pyelftools-0.32
"$ALPHA_CC" -g -gdwarf-4 -fno-eliminate-unused-debug-types -no-pie \
    "$ALPHA_REPO/libc_probe/headers.c" -o "$ALPHA_WORK/alpha-headers"
python3 "$ALPHA_REPO/libc_probe/generate_types.py" "$ALPHA_WORK/alpha-headers" \
    "$ALPHA_REPO/libc-alpha/src/unix/linux_like/linux/gnu/alpha/types.rs"
python3 "$ALPHA_REPO/libc_probe/generate_constants.py"
cp "$ALPHA_WORK/alpha-constants.c" "$ALPHA_REPO/libc_probe/constants.c"
python3 "$ALPHA_REPO/libc_probe/generate_checks.py"
python3 "$ALPHA_REPO/libc_probe/generate_layout_checks.py"
