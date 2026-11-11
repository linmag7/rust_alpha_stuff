#!/usr/bin/env bash
set -euo pipefail
w=$(cd -- "$(dirname -- "$0")" && pwd)
p=$(cd -- "$w/../.." && pwd)
cargo="$w/cargo-target/alphaev6-unknown-linux-gnu/release/cargo"
export TMPDIR="$w/tmp" CARGO_HOME="$w/cargo-native-home"
export RUSTUP_HOME="$p/work/upstream-20261111/rustup"
export QEMU_LD_PREFIX="$p/work/sdk/release/alpha-sdk/sysroot"
export LD_LIBRARY_PATH="$w/stage/opt/rust-alpha-experimental/lib"
mkdir -p "$CARGO_HOME"
"$cargo" --version --verbose
"$cargo" --list > "$w/logs/native-cargo-commands.txt"
if [ ! -f "$w/tests/native-cargo-project/Cargo.toml" ]; then
 "$cargo" new --vcs none "$w/tests/native-cargo-project"
fi
"$cargo" metadata --offline --no-deps --format-version=1 \
 --manifest-path "$w/tests/native-cargo-project/Cargo.toml" > "$w/logs/native-cargo-metadata.json"
python3 - "$w/logs/native-cargo-metadata.json" <<'CHECK'
import json,sys
m=json.load(open(sys.argv[1]));assert m['packages'][0]['name']=='native-cargo-project'
print('Native Alpha Cargo version/help/new/metadata: PASS')
CHECK
