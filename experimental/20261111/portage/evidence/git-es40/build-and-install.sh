#!/bin/bash
set -euo pipefail
base=/root/rust-alpha-git-2.56.1
cd "$base"
git --version > installed-git-before.txt
export FEATURES="$(portageq envvar FEATURES) -test"
export MAKEOPTS='-j4'
export PKGDIR="$base/binpkgs"
set +e
emerge --ignore-default-opts --ask=n --usepkg=n '=dev-vcs/git-2.56.1-r1::rust-alpha'
result=$?
printf '%s\n' "$result" > build-exit-status
exit "$result"
