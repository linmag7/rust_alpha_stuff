#!/usr/bin/env bash
# Run from the unpacked ABI/std bundle on an Alpha Linux machine.
set -u
ulimit -c 0
uname -a
getconf GNU_LIBC_VERSION
getconf PAGESIZE
status=0
run() {
    echo "$*"
    timeout 60 "$@"
    rc=$?
    echo "exit=$rc"
    if (( rc != 0 )); then status=1; fi
}
./c-constants > c-constants.txt || status=1
for profile in debug release; do
    for test in alpha_test abi_scalars abi_pair_arg abi_pair_ret abi_triple_arg abi_triple_callback abi_extended layouts runtime alpha_std_probe; do
        run "./$profile/$test"
    done
    if "./$profile/constants" > "constants-$profile.txt"; then
        run python3 ./compare_constants.py c-constants.txt "constants-$profile.txt"
    else
        status=1
    fi
done
# Independently compare the cross sysroot layouts to this machine's C headers.
if command -v gcc >/dev/null; then
    run gcc -O2 -no-pie layouts-dump.c -o native-layouts
    if [[ -x native-layouts ]]; then
        ./native-layouts > native-layouts.txt
        ./c-layouts > cross-layouts.txt
        run cmp cross-layouts.txt native-layouts.txt
    fi
    run gcc -O2 -no-pie constants.c -o native-constants
    if [[ -x native-constants ]]; then
        ./native-constants > native-constants.txt
        run python3 ./compare_constants.py native-constants.txt constants-release.txt
    fi
fi
exit "$status"
