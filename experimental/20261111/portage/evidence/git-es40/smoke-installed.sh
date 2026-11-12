#!/bin/bash
set -euo pipefail
base=/root/rust-alpha-git-2.56.1
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
/usr/bin/git version --build-options | tee "$base/installed-build-options.txt"
grep -Fx 'git version 2.56.1' "$base/installed-build-options.txt"
grep -Fx 'rust: enabled' "$base/installed-build-options.txt"
fixture=$(mktemp -d "$base/smoke.XXXXXX")
/usr/bin/git init --template= --initial-branch=main "$fixture/source"
cd "$fixture/source"
/usr/bin/git config user.name 'Alpha toolchain smoke test'
/usr/bin/git config user.email alpha-smoke@example.invalid
printf 'Alpha Rust Git λ\n' > hello.txt
/usr/bin/git add hello.txt
/usr/bin/git commit -m 'Initial smoke fixture'
/usr/bin/git switch -c feature
printf 'second line\n' >> hello.txt
/usr/bin/git commit -am 'Exercise branch and commit'
/usr/bin/git switch main
/usr/bin/git merge --ff-only feature
[[ $(/usr/bin/git rev-list --count HEAD) == 2 ]]
/usr/bin/git gc
/usr/bin/git fsck --strict
/usr/bin/git bundle create "$fixture/smoke.bundle" --all
/usr/bin/git bundle verify "$fixture/smoke.bundle"
/usr/bin/git clone "$fixture/smoke.bundle" "$fixture/clone"
cmp hello.txt "$fixture/clone/hello.txt"
/usr/bin/git -C "$fixture/clone" fsck --strict
[[ -z $(/usr/bin/git -C "$fixture/clone" status --porcelain) ]]
printf 'Installed Git smoke PASS: Rust enabled; init, commit, branch, merge, gc, fsck, bundle and clone\n'
portageq best_version / dev-vcs/git
