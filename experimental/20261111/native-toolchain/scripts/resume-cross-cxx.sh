#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$w/tmp"
export PATH="$w/../binutils-20261111/install/bin:$PATH"
cd "$w/build/cross-gcc"
make -j24 all-gcc all-target-libgcc all-target-libatomic all-target-libstdc++-v3 > "$w/logs/cross-cxx-build-resume.log" 2>&1
make install-gcc install-target-libgcc install-target-libatomic all-target-libstdc++-v3 > "$w/logs/cross-cxx-install.log" 2>&1
