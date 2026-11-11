#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
project=$(dirname -- "$repo")
rebase=$project/work/rebase-20261111
bundle=$rebase/build/es40-tests
export TMPDIR=$rebase/tmp
mkdir -p "$bundle/jit"
for profile in debug release; do
    mkdir -p "$bundle/$profile"
    for test in alpha_test abi_scalars abi_pair_arg abi_pair_ret abi_triple_arg abi_triple_callback abi_extended layouts runtime constants; do
        cp "$rebase/build/libc-isolated-validation/alphaev6-unknown-linux-gnu/$profile/$test" "$bundle/$profile/"
    done
    cp "$rebase/build/libc-isolated-std-validation/alphaev6-unknown-linux-gnu/$profile/alpha_std_probe" "$bundle/$profile/"
done
for iteration in {0..15}; do cp "$rebase/build/jit-smoke/jit-$iteration" "$bundle/jit/"; done
cp "$repo/scripts/run-fixed-tests-native.sh" "$bundle/run-libc-native.sh"
cp "$rebase/probes/libc_probe/"{layouts.c,layouts-dump.c,constants.c,compare_constants.py,unavailable-constants.json} "$bundle/"
cp "$rebase/build/c-constants" "$bundle/"
"$rebase/install/gcc-alpha/bin/alphaev6-unknown-linux-gnu-gcc" -O2 -no-pie \
    "$bundle/layouts-dump.c" -o "$bundle/c-layouts"
cat > "$bundle/run-native.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
bash ./run-libc-native.sh
for iteration in {0..15}; do timeout 30 "./jit/jit-$iteration" "$iteration"; done
printf '%s\n' 'GCC 17/libgccjit: PASS (16 contexts, 1024 runtime checks)' 'Rebased component validation: PASS'
RUN
chmod +x "$bundle/run-native.sh"
tar -czf "$rebase/es40-rebased-components.tar.gz" -C "$bundle" .
printf '%s\n' "$rebase/es40-rebased-components.tar.gz"
