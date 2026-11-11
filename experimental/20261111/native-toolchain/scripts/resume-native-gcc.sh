#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
export TMPDIR="$w/tmp" PATH="$w/cross/bin:$p/work/binutils-20261111/install/bin:$PATH"
cd "$w/build/native-gcc"
make -j24 all-host all-target "CXX_FOR_TARGET=$w/cross/bin/alphaev6-unknown-linux-gnu-g++ -nostdinc++" 'EXTRA_BUILD_FLAGS=CFLAGS="-O2 -g0" CXXFLAGS="-O2 -g0" LDFLAGS=""' > "$w/logs/native-gcc-build.log" 2>&1
make DESTDIR="$w/stage" install > "$w/logs/native-gcc-install.log" 2>&1
