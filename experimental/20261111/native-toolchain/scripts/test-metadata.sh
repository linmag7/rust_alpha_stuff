#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
stage=${1:-$w/host-rust-build/x86_64-unknown-linux-gnu/stage2}
backend=${2:-$w/host-backend-target/release/librustc_codegen_gcc.so}
export TMPDIR="$w/tmp" RUSTC_BOOTSTRAP=1 LD_LIBRARY_PATH="$stage/lib"
export CG_GCCJIT_LIBRARY_PATH="$p/work/rebase-20261111/build/gcc/gcc/libgccjit.so"
"$stage/bin/rustc" "$w/tests/metadata.rs" --crate-type=dylib --target alphaev6-unknown-linux-gnu \
 -Zcodegen-backend="$backend" -Cpanic=abort -Clinker="$w/alpha-link" -o "$w/tests/libmetadata.so"
"$p/work/binutils-20261111/install/bin/alphaev6-unknown-linux-gnu-readelf" -h -S "$w/tests/libmetadata.so"
"$stage/bin/rustc" "$w/tests/metadata-consumer.rs" --crate-type=rlib --target alphaev6-unknown-linux-gnu \
 -Zcodegen-backend="$backend" -Cpanic=abort --emit=metadata \
 --extern "metadata=$w/tests/libmetadata.so" -o "$w/tests/metadata-consumer.rmeta"
python3 - "$w/tests/libmetadata.so" <<'CHECK'
from pathlib import Path
import sys
b=Path(sys.argv[1]).read_bytes()
assert b[:6] == b'\x7fELF\x02\x01'
assert int.from_bytes(b[18:20], 'little') == 0x9026
print('Alpha ELF dylib metadata write/read: PASS')
CHECK
