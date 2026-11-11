#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$here/../../../.." && pwd)
w="$p/work/portage-20261111"
export PORTAGE_OVERRIDE_EPREFIX="" PORTAGE_CONFIGROOT="$w/config" PORTAGE_PROFILE="$w/test-master/profiles/test-alpha"
export ROOT="$w/root" TMPDIR="$w/tmp"
if [[ ${1:-} == emerge ]]; then
 [[ " $* " == *" --pretend "* ]] || { echo 'Test harness permits emerge --pretend only' >&2; exit 2; }
 # Avoid requiring a second, host Gentoo profile for this read-only resolver check.
 export ROOT=/
fi
for arg in "$@"; do
 case $arg in merge|qmerge|unmerge) echo 'Test harness does not install packages' >&2; exit 2 ;; esac
done
export PATH="$w/host-tools:$w/venv/bin:$PATH"
export QEMU_LD_PREFIX="$p/work/sdk/release/alpha-sdk/sysroot" ALPHA_SYSROOT="$p/work/sdk/release/alpha-sdk/sysroot"
exec "$w/venv/bin/$@"
