#!/usr/bin/env bash
set -euo pipefail
work=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$work/../.." && pwd)
dejagnu="$project/work/upstream-20261111/dejagnu"
export PATH="$dejagnu/usr/bin:$PATH"
export LD_LIBRARY_PATH="$dejagnu/usr/lib/x86_64-linux-gnu"
export DEJAGNU=/dev/null DEJAGNULIBS="$dejagnu/usr/share/dejagnu" TMPDIR="$work/tmp"
make -C "$work/build" -j3 check-gas check-ld check-binutils \
  CC_FOR_TARGET="$work/alpha-cc" CXX_FOR_TARGET="$work/alpha-cc" \
  > "$work/logs/check.log" 2>&1
