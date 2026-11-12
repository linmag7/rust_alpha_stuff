#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mode=${1:---check}
case $mode in --check|--install|--update) ;; *) echo 'Usage: setup-es40.sh --check|--install|--update'; exit 2;; esac
[[ $(uname -m) == alpha ]] || { echo 'Run on native Alpha Gentoo'; exit 1; }
command -v portageq >/dev/null
sha256sum --quiet -c SHA256SUMS
v=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
[[ $(printf '2.39\n%s\n' "$v" | sort -V | head -n1) == 2.39 ]] || { echo 'glibc >=2.39 required; tested on2.41'; exit 1; }
echo "Alpha Gentoo, glibc $v, CHOST=$(portageq envvar CHOST). Bundle checksums PASS."
[[ $mode != --check ]] || exit 0
[[ $EUID == 0 ]] || { echo 'Root required'; exit 1; }
repo=/var/db/repos/rust-alpha
for d in repos.conf package.accept_keywords package.unmask package.use profile/package.use.mask; do
 [[ ! -f /etc/portage/$d ]] || { echo "Merge config manually: /etc/portage/$d is a file"; exit 1; }
done
pairs=('rust-alpha.conf:repos.conf/rust-alpha.conf' 'package.accept_keywords:package.accept_keywords/rust-alpha-20261112' 'package.unmask:package.unmask/rust-alpha-20261112' 'package.use:package.use/rust-alpha-20261112' 'package.use.mask:profile/package.use.mask/rust-alpha-20261112')
if [[ $mode == --install ]]; then
 [[ ! -e $repo ]] || diff -qr overlay "$repo" >/dev/null || { echo 'Existing overlay differs; review changes and use --update'; exit 1; }
 for pair in "${pairs[@]}"; do
  dst=/etc/portage/${pair#*:}
  [[ ! -e $dst ]] || cmp -s "config/${pair%%:*}" "$dst" || { echo "Existing config differs: $dst"; exit 1; }
 done
fi
distdir=$(portageq envvar DISTDIR)
[[ $distdir == /* && -n $distdir ]] || exit 1
for f in distfiles/*; do
 [[ ! -e $distdir/${f##*/} ]] || cmp -s "$f" "$distdir/${f##*/}" || { echo "Different existing distfile: ${f##*/}"; exit 1; }
done
if [[ $mode == --update ]]; then
 backup=/var/backups/rust-alpha-$(date +%Y%m%d-%H%M%S)-$$
 mkdir -p "$backup"
 [[ ! -d $repo ]] || cp -a "$repo" "$backup/overlay"
 cp -a /etc/portage "$backup/portage-config"
 echo "Previous overlay/config backed up in $backup"
fi
mkdir -p "$repo" "$distdir"
cp -a overlay/. "$repo/"
for pair in "${pairs[@]}"; do
 dst=/etc/portage/${pair#*:};mkdir -p "$(dirname "$dst")"
 install -m644 "config/${pair%%:*}" "$dst"
done
for f in distfiles/*; do [[ -e $distdir/${f##*/} ]] || install -m644 "$f" "$distdir/${f##*/}"; done
echo 'Overlay/config/distfiles installed. No packages merged.'
echo 'Next: emerge --ask =dev-lang/rust-alpha-bin-1.102.0_pre20261111-r1::rust-alpha'
