#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
work=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$work/../.." && pwd)
dejagnu="$project/work/upstream-20261111/dejagnu"
sdk="$project/work/sdk/release/alpha-sdk"
export PATH="$dejagnu/usr/bin:$PATH" LD_LIBRARY_PATH="$dejagnu/usr/lib/x86_64-linux-gnu"
export DEJAGNU=/dev/null DEJAGNULIBS="$dejagnu/usr/share/dejagnu" TMPDIR="$work/tmp"
mkdir -p "$work/old-check"
cp "$work/build/ld/site.exp" "$work/old-check/site.exp"
cd "$work/old-check"
for tool in as ld nm objdump objcopy ar strip size readelf ranlib; do
  printf 'set %s {%s}\n' "$tool" "$sdk/gcc/bin/alphaev6-unknown-linux-gnu-$tool" >> site.exp
done
printf 'set ld_testsuite_bindir {%s}\n' "$sdk/gcc/alphaev6-unknown-linux-gnu/bin" >> site.exp
runtest --tool ld --srcdir "$work/src/ld/testsuite" \
  "CC_FOR_TARGET=$sdk/bin/alpha-cc" "CXX_FOR_TARGET=$sdk/bin/alpha-cc" \
  "LD=$sdk/gcc/bin/alphaev6-unknown-linux-gnu-ld" "READELF=$sdk/gcc/bin/alphaev6-unknown-linux-gnu-readelf" \
  > "$work/logs/check-baseline.log" 2>&1
