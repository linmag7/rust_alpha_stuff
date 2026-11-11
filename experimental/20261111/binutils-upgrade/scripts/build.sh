#!/usr/bin/env bash
set -euo pipefail
work=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$work/../.." && pwd)
export TMPDIR="$work/tmp"
mkdir -p "$work/build" "$work/install" "$work/logs" "$TMPDIR"
test "$(git -C "$work/src" rev-parse HEAD)" = "$(cat "$work/revision")"
cd "$work/build"
"$work/src/configure" --target=alphaev6-unknown-linux-gnu \
  --prefix="$work/install" --with-sysroot="$project/work/sdk/release/alpha-sdk/sysroot" \
  --disable-gdb --disable-gdbserver --disable-sim --disable-gprofng \
  --disable-nls --disable-werror > "$work/logs/configure.log" 2>&1
make -j24 > "$work/logs/build.log" 2>&1
make install > "$work/logs/install.log" 2>&1
"$work/install/bin/alphaev6-unknown-linux-gnu-ld" --version
