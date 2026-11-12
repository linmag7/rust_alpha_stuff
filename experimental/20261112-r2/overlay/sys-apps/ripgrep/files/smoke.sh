#!/usr/bin/env bash
set -euo pipefail
binary=$1
fixture=$2
mkdir -p "$fixture"
printf 'alpha\nbeta\nalpha λ\n' > "$fixture/input.txt"
[[ $("$binary" --no-config -n alpha "$fixture/input.txt") == $'1:alpha\n3:alpha λ' ]]
[[ $("$binary" --no-config -o '\p{Greek}+' "$fixture/input.txt") == λ ]]
if "$binary" --no-config absent "$fixture/input.txt"; then exit 1; else [[ $? == 1 ]]; fi
mkdir -p "$fixture/tree"
printf 'needle\n' > "$fixture/tree/keep.txt"
printf 'needle\n' > "$fixture/tree/skip.txt"
printf 'skip.txt\n' > "$fixture/tree/.gitignore"
[[ $("$binary" --no-config --no-require-git -l needle "$fixture/tree") == "$fixture/tree/keep.txt" ]]
"$binary" --generate man > "$fixture/rg.1"
[[ -s "$fixture/rg.1" ]]
printf 'ripgrep: search, Unicode, exit status, ignore rules and man generation PASS\n'
