#!/usr/bin/env bash
# Host-only packaging helper: fake ownership inside the private staging tree.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$here/../../../.." && pwd)
w="$p/work/portage-20261111"
ebuild=$(realpath -- "$1")
atom=${ebuild#"$here/overlay/"}
category=${atom%%/*}
cpv="$category/$(basename -- "$ebuild" .ebuild)"
image="$w/portage-tmp/portage/$cpv/image"
[[ -d $image && $ebuild == "$here/overlay/"* ]] || exit 2
rm -f "$w/portage-tmp/portage/$cpv/.packaged"
fakeroot bash -c '
 set -euo pipefail
 export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libfakeroot/libfakeroot-sysv.so
 chown -R 0:0 "$1"
 exec "$2/test-portage.sh" ebuild "$3" package
' bash "$image" "$here" "$ebuild"
