#!/usr/bin/env bash
set -euo pipefail
binary=$1
fixture=$2
mkdir -p "$fixture/tree/nested"
printf 'one\n' > "$fixture/tree/first.txt"
printf 'two\n' > "$fixture/tree/nested/second.txt"
printf 'hidden\n' > "$fixture/tree/.hidden.txt"
printf 'data\n' > "$fixture/tree/other.dat"
actual=$("$binary" --base-directory "$fixture/tree" --color never --extension txt | sort)
[[ $actual == $'first.txt\nnested/second.txt' ]]
[[ $("$binary" --base-directory "$fixture/tree" --hidden --color never --extension txt | wc -l) == 3 ]]
"$binary" --gen-completions bash > "$fixture/fd.bash"
[[ -s "$fixture/fd.bash" ]]
printf 'fd: recursive search, extension filter, hidden files and completion generation PASS\n'
