# Run with fish --no-config; temporary files remain within campaign directory.
set -l base /root/rust-alpha-overnight-20261111/fish-smoke-work
mkdir -p $base; or exit 1
function assert_equal
    if test "$argv[1]" != "$argv[2]"
        printf 'FAIL: %s != %s\n' "$argv[1]" "$argv[2]" >&2
        exit 1
    end
end
assert_equal (math '6 * 7') 42
assert_equal (printf 'alpha\nbeta\n' | string match 'b*') beta
printf 'redirect works\n' > $base/output.txt; or exit 1
assert_equal (cat $base/output.txt) 'redirect works'
set -l total 0
for n in 1 2 3 4
    set total (math $total + $n)
end
assert_equal $total 10
/bin/sh -c 'exit 7'
assert_equal $status 7
/bin/sleep 1 &
set -l child $last_pid
wait $child; or exit 1
set -l result (command /usr/bin/printf '%s' 'external command')
assert_equal "$result" 'external command'
printf 'Fish pipelines, redirection, loop, exit status, background wait: PASS\n'
