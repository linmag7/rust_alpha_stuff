#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$w/tmp"
mkdir -p "$w/dist"
archive=alpha-native-toolchain-20261111-experimental.tar.xz
tar -C "$w/stage" -c opt/rust-alpha-experimental | xz -T4 -3 > "$w/dist/$archive"
xz -t "$w/dist/$archive"
(cd "$w/dist" && sha256sum "$archive" > "$archive.sha256")
