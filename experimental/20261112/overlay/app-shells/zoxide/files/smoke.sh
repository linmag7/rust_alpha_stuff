#!/usr/bin/env bash
set -euo pipefail
binary=$1
fixture=$2
mkdir -p "$fixture/data" "$fixture/project-alpha" "$fixture/project-beta"
export _ZO_DATA_DIR="$fixture/data"
"$binary" add "$fixture/project-alpha" "$fixture/project-beta"
[[ $("$binary" query project-alpha) == "$fixture/project-alpha" ]]
[[ $("$binary" query --list | wc -l) == 2 ]]
"$binary" remove "$fixture/project-beta"
if "$binary" query project-beta; then exit 1; else [[ $? == 1 ]]; fi
"$binary" init bash > "$fixture/init.bash"
[[ -s "$fixture/init.bash" ]]
printf 'zoxide: persistent database, add/query/remove and shell initialization PASS\n'
