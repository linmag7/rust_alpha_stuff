#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
native="$w/stage/opt/rust-alpha-experimental"
sysroot="$p/work/sdk/release/alpha-sdk/sysroot"
export TMPDIR="$w/tmp" QEMU_LD_PREFIX="$sysroot" ALPHA_SYSROOT="$sysroot"
export PATH="$native/bin:$PATH"
export GCC_EXEC_PREFIX="$native/lib/gcc/"
export COMPILER_PATH="$native/libexec/gcc/alphaev6-unknown-linux-gnu/17.0.0:$native/bin"
export LIBRARY_PATH="$native/lib:$native/lib/gcc/alphaev6-unknown-linux-gnu/17.0.0:$sysroot/usr/lib"
export LD_LIBRARY_PATH="$native/lib"
"$native/bin/gcc" -v > "$w/logs/native-gcc-version.log" 2>&1
"$native/bin/gcc" --sysroot="$sysroot" -O2 -mieee "$w/tests/hello.c" -o "$w/tests/native-hello"
"$w/tests/native-hello"
"$native/bin/g++" --sysroot="$sysroot" -O2 -mieee "$w/tests/cxx.cc" -o "$w/tests/native-cxx"
"$w/tests/native-cxx"
"$native/bin/gcc" --sysroot="$sysroot" -O2 -mieee -rdynamic -I"$native/include" \
 "$w/tests/jit-native.c" -L"$native/lib" -lgccjit -o "$w/tests/jit-native"
"$w/tests/jit-native"
