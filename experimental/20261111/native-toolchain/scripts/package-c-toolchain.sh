#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$w/tmp"
mkdir -p "$w/dist"
tar -C "$w/stage" -c opt/rust-alpha-experimental | xz -T4 -3 > "$w/dist/alpha-native-gcc-binutils-20261111.tar.xz"
(cd "$w/dist" && sha256sum alpha-native-gcc-binutils-20261111.tar.xz > alpha-native-gcc-binutils-20261111.tar.xz.sha256)
