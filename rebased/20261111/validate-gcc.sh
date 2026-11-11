#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$here/../../.." && pwd)
rebase=$project/work/rebase-20261111
prefix=$rebase/install/gcc-alpha
out=$rebase/build/jit-smoke
export TMPDIR=$rebase/tmp
export PATH=$prefix/bin:$PATH
mkdir -p "$out"
gcc -O2 -Wall -Wextra -Werror "$here/jit-smoke.c" -I"$prefix/include" \
    -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -lgccjit -o "$out/generate"
"$out/generate" "$out"
for iteration in {0..15}; do
    "$prefix/bin/alphaev6-unknown-linux-gnu-gcc" -O2 -Wall -Wextra -Werror -no-pie \
        "$here/jit-driver.c" "$out/jit-$iteration.o" -o "$out/jit-$iteration"
    "$here/../../scripts/alpha-run.sh" "$out/jit-$iteration" "$iteration"
done
printf '%s\n' 'GCC 17/libgccjit: PASS (16 contexts, 1024 runtime checks)'
