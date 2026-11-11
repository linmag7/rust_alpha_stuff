#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
export TMPDIR="$w/tmp"
export CC="$p/work/binutils-20261111/alpha-cc" CXX=false
export AR="$p/work/binutils-20261111/install/bin/alphaev6-unknown-linux-gnu-ar"
export RANLIB="$p/work/binutils-20261111/install/bin/alphaev6-unknown-linux-gnu-ranlib"
export CFLAGS='-O2 -g0 -mcpu=ev6 -mieee'
mkdir -p "$w/build/binutils"
cd "$w/build/binutils"
"$p/work/binutils-20261111/src/configure" --build=x86_64-pc-linux-gnu \
 --host=alphaev6-unknown-linux-gnu --target=alphaev6-unknown-linux-gnu \
 --prefix=/opt/rust-alpha-experimental --disable-gdb --disable-gdbserver \
 --disable-sim --disable-gprofng --disable-nls --disable-werror \
 --without-zstd --without-debuginfod --without-system-zlib \
 > "$w/logs/binutils-configure.log" 2>&1
make -j24 > "$w/logs/binutils-build.log" 2>&1
make DESTDIR="$w/stage" install > "$w/logs/binutils-install.log" 2>&1
