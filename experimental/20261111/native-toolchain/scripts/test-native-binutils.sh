#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
dejagnu="$p/work/upstream-20261111/dejagnu"
export TMPDIR="$w/tmp" QEMU_LD_PREFIX="$p/work/sdk/release/alpha-sdk/sysroot"
export PATH="$dejagnu/usr/bin:$PATH" LD_LIBRARY_PATH="$dejagnu/usr/lib/x86_64-linux-gnu"
export DEJAGNU=/dev/null DEJAGNULIBS="$dejagnu/usr/share/dejagnu"
mkdir -p "$w/native-binutils-tests"
cd "$w/native-binutils-tests"
cp "$p/work/binutils-20261111/build/ld/site.exp" site.exp
for tool in as ld nm objdump objcopy ar strip size readelf ranlib; do
 printf 'set %s {%s}\n' "$tool" "$w/stage/opt/rust-alpha-experimental/bin/$tool" >> site.exp
done
printf 'set ld_testsuite_bindir {%s}\n' "$w/stage/opt/rust-alpha-experimental/bin" >> site.exp
runtest --tool ld --srcdir "$p/work/binutils-20261111/src/ld/testsuite" \
 "LD=$w/stage/opt/rust-alpha-experimental/bin/ld" \
 "READELF=$w/stage/opt/rust-alpha-experimental/bin/readelf" \
 "CC_FOR_TARGET=$p/work/binutils-20261111/alpha-cc" ld-alpha/alpha.exp \
 > "$w/logs/native-binutils-tests.log" 2>&1
