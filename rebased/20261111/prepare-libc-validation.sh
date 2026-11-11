#!/usr/bin/env bash
# Keep the known-good compiler and its standard-library lockfile unchanged.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
project=$(dirname -- "$repo")
rebase=$project/work/rebase-20261111
baseline=$project/work/rust-build/x86_64-unknown-linux-gnu/stage1
isolated=$rebase/install/validated-rust
export TMPDIR=$rebase/tmp
export CARGO_HOME=$project/work/cargo-home
mkdir -p "$rebase/src/validated-std" "$rebase/probes" "$rebase/logs" "$TMPDIR"
if [[ ! -d $isolated ]]; then cp -as "$baseline" "$isolated"; fi
if [[ ! -d $rebase/src/validated-std/library ]]; then
    cp -a --reflink=auto "$project/work/rust-src/library" "$rebase/src/validated-std/"
fi
ln -sfn "$rebase/src/validated-std" "$isolated/lib/rustlib/src/rust"
python3 - "$baseline/bin/rustc" "$isolated/bin/rustc" "$repo" "$rebase" <<'PY'
from pathlib import Path
import shlex
import shutil
import sys
original, wrapper, repo, rebase = map(Path, sys.argv[1:])
# Current Rust already casts this flag; backport only for the old validation std.
process = rebase/'src/validated-std/library/std/src/sys/process/unix/unix.rs'
process.write_text(process.read_text().replace(
    'flags |= libc::POSIX_SPAWN_SETSID;',
    'flags |= libc::POSIX_SPAWN_SETSID as i32;'))
wrapper.unlink(missing_ok=True)
wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
sysroot=$(cd -- "$(dirname -- "$0")/.." && pwd)
explicit=false
for argument in "$@"; do
    case "$argument" in --sysroot|--sysroot=*) explicit=true;; esac
done
if [[ $explicit == false ]]; then set -- --sysroot "$sysroot" "$@"; fi
exec ''' + shlex.quote(str(original)) + ''' "$@"
''')
wrapper.chmod(0o755)
for name in ('libc_probe', 'std_probe', 'abi_probe', 'alpha_test'):
    destination = rebase/'probes'/name
    if not destination.exists():
        shutil.copytree(repo/name, destination)
    manifest = destination/'Cargo.toml'
    manifest.write_text(manifest.read_text().replace('../libc-alpha', '../../src/libc'))
PY
ln -sfn /home/lindholm/x-tools/alphaev6-unknown-linux-gnu/lib/libgccjit.so \
    "$isolated/lib/rustlib/x86_64-unknown-linux-gnu/codegen-backends/lib/alphaev6-unknown-linux-gnu/libgccjit.so"
export RUSTC=$isolated/bin/rustc
"$isolated/bin/cargo" update --offline --manifest-path "$rebase/src/validated-std/library/Cargo.toml" \
    -p libc --precise 0.2.191 --config "patch.crates-io.libc.path=\"$rebase/src/libc\""
