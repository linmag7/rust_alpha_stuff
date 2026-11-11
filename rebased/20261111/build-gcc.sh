#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
project=$(dirname -- "$repo")
rebase=$project/work/rebase-20261111
cross=${ALPHA_BASE_CROSS:-/home/lindholm/x-tools/alphaev6-unknown-linux-gnu}
export TMPDIR=$rebase/tmp
export PATH="$cross/bin:$PATH"
mkdir -p "$TMPDIR" "$rebase/build/gcc" "$rebase/logs"
cd "$rebase/build/gcc"
if [[ ! -f Makefile ]]; then
    "$rebase/src/gcc/configure" --target=alphaev6-unknown-linux-gnu \
        --prefix="$rebase/install/gcc-alpha" \
        --with-sysroot="$cross/alphaev6-unknown-linux-gnu/sysroot" \
        --with-as="$cross/bin/alphaev6-unknown-linux-gnu-as" \
        --with-ld="$cross/bin/alphaev6-unknown-linux-gnu-ld" \
        --enable-languages=c,jit --enable-host-shared --enable-shared \
        --enable-threads=posix --with-cpu=ev6 --with-tune=ev6 \
        --disable-bootstrap --disable-multilib --disable-nls \
        --disable-libsanitizer --disable-libquadmath --disable-libgomp \
        --disable-libssp --disable-libstdcxx --disable-werror \
        --without-isl --enable-checking=release > "$rebase/logs/gcc-configure.log" 2>&1
fi
make -j"${ALPHA_JOBS:-24}" all-gcc all-target-libgcc > "$rebase/logs/gcc-build.log" 2>&1
make install-gcc install-target-libgcc > "$rebase/logs/gcc-install.log" 2>&1
# Reuse the saved cross-binutils alongside the newly built compiler.
for tool in ar as ld nm objcopy objdump ranlib readelf size strings strip; do
    ln -sfn "$cross/bin/alphaev6-unknown-linux-gnu-$tool" \
        "$rebase/install/gcc-alpha/bin/alphaev6-unknown-linux-gnu-$tool"
done
