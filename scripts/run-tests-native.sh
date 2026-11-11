#!/usr/bin/env bash
# Run inside an unpacked bundle containing debug/ and release/ binaries.
set -u
ulimit -c 0
uname -a
getconf GNU_LIBC_VERSION
getconf PAGESIZE
status=0
for profile in debug release; do
    for test in alpha_test abi_scalars abi_pair_arg abi_pair_ret abi_triple_arg abi_triple_callback; do
        printf '\n%s/%s\n' "$profile" "$test"
        timeout 30 "./$profile/$test"
        rc=$?
        printf 'exit=%s\n' "$rc"
        if (( rc != 0 )); then status=1; fi
    done
done
exit "$status"
