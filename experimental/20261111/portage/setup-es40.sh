#!/usr/bin/env bash
# Run on the ES40 after transferring/extracting the completed bundle.
set -euo pipefail
cd -- "$(dirname -- "$0")"
mode=${1:---check}
[[ $mode == --check || $mode == --install ]] || { echo 'Usage: ./setup-es40.sh [--check|--install]' >&2; exit 2; }
case $(uname -m) in alpha*) ;; *) echo 'Run this setup on the Alpha machine, not the workstation.' >&2; exit 1 ;; esac
command -v portageq >/dev/null || { echo 'Gentoo Portage is required.' >&2; exit 1; }
glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
[[ $(printf '2.39\n%s\n' "$glibc" | sort -V | head -n1) == 2.39 ]] || { echo "This package set requires glibc >=2.39; found $glibc" >&2; exit 1; }
chost=$(portageq envvar CHOST)
printf 'Machine: %s\nglibc: %s\nCHOST: %s\n' "$(uname -m)" "$glibc" "$chost"
if [[ -f binpkgs/Packages ]]; then
 expected=$(sed -n 's/^CHOST: //p' binpkgs/Packages | head -n1)
 if [[ $chost != "$expected" ]]; then
  printf 'Binary packages were prepared for CHOST=%s. Use source ebuilds until binary compatibility is checked.\n' "$expected"
 else
  printf 'Binary package CHOST matches; runtime testing is still required.\n'
 fi
else
 printf 'Source-build install kit: applications will compile natively through their ebuilds.\n'
fi
printf 'Portage temporary directory: %s\n' "$(portageq envvar PORTAGE_TMPDIR)"
[[ $mode == --install ]] || exit 0
[[ $EUID == 0 ]] || { echo '--install must be run as root.' >&2; exit 1; }
sha256sum --quiet -c SHA256SUMS
repo=/var/db/repos/rust-alpha
for dir in /etc/portage/repos.conf /etc/portage/package.accept_keywords /etc/portage/package.unmask; do
 [[ ! -f $dir ]] || { echo "$dir is a file; merge the supplied configuration manually." >&2; exit 1; }
done
if [[ -e $repo ]]; then
 diff -qr overlay "$repo" >/dev/null || { echo "$repo already exists with different contents; review it before updating." >&2; exit 1; }
fi
for pair in 'config/rust-alpha.conf:/etc/portage/repos.conf/rust-alpha.conf' 'config/package.accept_keywords:/etc/portage/package.accept_keywords/rust-alpha' 'config/package.unmask:/etc/portage/package.unmask/rust-alpha'; do
 src=${pair%%:*}; dst=${pair#*:}
 [[ ! -e $dst ]] || cmp -s "$src" "$dst" || { echo "Refusing to overwrite differing $dst" >&2; exit 1; }
done
install -d "$repo" /etc/portage/repos.conf /etc/portage/package.accept_keywords /etc/portage/package.unmask
cp -a overlay/. "$repo/"
install -m644 config/rust-alpha.conf /etc/portage/repos.conf/rust-alpha.conf
install -m644 config/package.accept_keywords /etc/portage/package.accept_keywords/rust-alpha
install -m644 config/package.unmask /etc/portage/package.unmask/rust-alpha
distdir=$(portageq envvar DISTDIR)
[[ -n $distdir && $distdir == /* ]] || { echo 'Invalid Portage DISTDIR' >&2; exit 1; }
install -d "$distdir"
for archive in distfiles/*; do
 name=$(basename -- "$archive")
 if [[ -e $distdir/$name ]]; then
  cmp -s "$archive" "$distdir/$name" || { echo "Different existing distfile: $distdir/$name" >&2; exit 1; }
 else
  install -m644 "$archive" "$distdir/$name"
 fi
done
printf 'Overlay and distfiles installed. No compiler or application has been merged yet.\n'
printf 'Next: emerge --ask dev-lang/rust-alpha-bin::rust-alpha\n'
printf 'Then: emerge --ask sys-apps/ripgrep::rust-alpha sys-apps/fd::rust-alpha app-shells/zoxide::rust-alpha\n'
