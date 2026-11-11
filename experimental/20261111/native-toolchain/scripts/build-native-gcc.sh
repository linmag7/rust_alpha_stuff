#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
export TMPDIR="$w/tmp"
export PATH="$w/cross/bin:$p/work/binutils-20261111/install/bin:$PATH"
export CC="$w/cross/bin/alphaev6-unknown-linux-gnu-gcc"
export CXX="$w/cross/bin/alphaev6-unknown-linux-gnu-g++"
export AR="$p/work/binutils-20261111/install/bin/alphaev6-unknown-linux-gnu-ar"
export RANLIB="$p/work/binutils-20261111/install/bin/alphaev6-unknown-linux-gnu-ranlib"
export CFLAGS='-O2 -g0 -mcpu=ev6 -mieee'
export CXXFLAGS='-O2 -g0 -mcpu=ev6 -mieee'
export CFLAGS_FOR_BUILD='-O2 -g0' CXXFLAGS_FOR_BUILD='-O2 -g0'
mkdir -p "$w/build/native-gcc"
cd "$w/build/native-gcc"
"$p/work/rebase-20261111/src/gcc/configure" \
 --build=x86_64-pc-linux-gnu --host=alphaev6-unknown-linux-gnu --target=alphaev6-unknown-linux-gnu \
 --prefix=/opt/rust-alpha-experimental --enable-languages=c,c++,jit,lto \
 --with-build-time-tools="$p/work/binutils-20261111/install/alphaev6-unknown-linux-gnu/bin" \
 --with-sysroot=/ --with-build-sysroot="$p/work/sdk/release/alpha-sdk/sysroot" \
 --with-gmp="$w/deps" --with-mpfr="$w/deps" --with-mpc="$w/deps" \
 --with-host-libstdcxx='-static-libstdc++ -static-libgcc' \
 --enable-host-shared --enable-shared --enable-threads=posix --with-cpu=ev6 --with-tune=ev6 \
 --disable-bootstrap --disable-multilib --disable-nls --disable-libsanitizer \
 --disable-libquadmath --disable-libgomp --disable-libssp --disable-werror \
 --without-isl --enable-checking=release \
 CC_FOR_BUILD=gcc CXX_FOR_BUILD=g++ \
 CC_FOR_TARGET="$CC" CXX_FOR_TARGET="$CXX -nostdinc++" \
 > "$w/logs/native-gcc-configure.log" 2>&1
make -j24 all-host all-target "CXX_FOR_TARGET=$w/cross/bin/alphaev6-unknown-linux-gnu-g++ -nostdinc++" 'EXTRA_BUILD_FLAGS=CFLAGS="-O2 -g0" CXXFLAGS="-O2 -g0" LDFLAGS=""' > "$w/logs/native-gcc-build.log" 2>&1
make DESTDIR="$w/stage" install > "$w/logs/native-gcc-install.log" 2>&1
