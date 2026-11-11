#!/bin/bash
set -euo pipefail
cd /root/rust-alpha-install-20261111
sha256sum -c rust-alpha-portage-20261111.tar.sha256
tar -xf rust-alpha-portage-20261111.tar
cd rust-alpha-portage
./setup-es40.sh --check
sha256sum --quiet -c SHA256SUMS
for recipe in 'sys-apps ripgrep rg' 'sys-apps fd fd' 'app-shells zoxide zoxide'; do
 read -r category name binary <<< "$recipe"
 mkdir -p "hardware-check/$name"
 package=$(find "binpkgs/$category/$name" -name '*.gpkg.tar' -print -quit)
 tar -xOf "$package" --wildcards '*/image.tar.xz' | tar -xJf - -C "hardware-check/$name"
 executable="$PWD/hardware-check/$name/image/usr/bin/$binary"
 "$executable" --version
 bash "overlay/$category/$name/files/smoke.sh" "$executable" "$PWD/hardware-check/$name/fixtures"
done
echo 'ES40 application payload tests PASS'
