#!/usr/bin/env bash
# Rebuild the isolated compiler and matching backend from the preserved sources.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
project=${ALPHA_PROJECT:-$(dirname -- "$repo")}
work=${ALPHA_WORK:-$project/work}
installed=${ALPHA_STAGE0:-/home/lindholm/toolchains/rust-git-dev}
llvm=$project/rust/build/x86_64-unknown-linux-gnu/ci-llvm
stage=$work/rust-build/x86_64-unknown-linux-gnu/stage1
export TMPDIR=$work/tmp CARGO_HOME=$work/cargo-home CARGO_NET_OFFLINE=true
mkdir -p "$TMPDIR" "$work/logs"
if [[ ! -d $work/rust-src ]]; then
    mkdir -p "$work/rust-src"
    rsync -a --exclude=.git --exclude=build --exclude=target --exclude=/rust --exclude=/src/llvm-project/ "$project/rust/" "$work/rust-src/"
    patch -d "$work/rust-src" -p1 < "$repo/patches/rust-alpha-abi-std.patch"
fi
cat > "$work/rust-src/bootstrap.toml" <<CONFIG
change-id = "ignore"
[llvm]
link-shared = true
download-ci-llvm = false
[build]
build-dir = "$work/rust-build"
rustc = "$installed/bin/rustc"
cargo = "$installed/bin/cargo"
extended = false
submodules = false
docs = false
[rust]
llvm-tools = false
download-rustc = false
channel = "dev"
[target.x86_64-unknown-linux-gnu]
llvm-config = "$llvm/bin/llvm-config"
llvm-has-rust-patches = true
CONFIG
export LD_LIBRARY_PATH=$installed/lib:$llvm/lib
if [[ ${1:-} != --finish ]]; then
    abi_source_before=$(sha256sum "$work/rust-src/compiler/rustc_target/src/callconv/alpha.rs")
    (cd "$work/rust-src" && python3 x.py build --stage 1 compiler/rustc --jobs "${ALPHA_JOBS:-24}")
    [[ $(sha256sum "$work/rust-src/compiler/rustc_target/src/callconv/alpha.rs") == "$abi_source_before" ]] || { echo "Bootstrap overwrote the ABI source" >&2; exit 1; }
fi
# x.py's local-rebuild path does not install the rustc-dev metadata required by
# an out-of-tree codegen backend. Supply the matching artifacts and host std.
hostlib=$stage/lib/rustlib/x86_64-unknown-linux-gnu/lib
mkdir -p "$hostlib" "$stage/lib/rustlib/src"
cp "$work/rust-build/x86_64-unknown-linux-gnu/stage1-rustc/x86_64-unknown-linux-gnu/release/deps/"*.{rmeta,rlib,so} "$hostlib/"
cp "$work/rust-build/x86_64-unknown-linux-gnu/stage1-rustc/release/deps/"*.so "$hostlib/"
while IFS= read -r record; do
    [[ $record == file:* ]] || continue
    file=${record#file:}
    relative=${file#"$installed/"}
    [[ $relative != "$file" ]] || { echo "Unexpected std manifest path: $file" >&2; exit 1; }
    mkdir -p "$stage/$(dirname -- "$relative")"
    cp "$file" "$stage/$relative"
done < "$installed/lib/rustlib/manifest-rust-std-x86_64-unknown-linux-gnu"
ln -sfn "$llvm/lib/libLLVM.so.21.1-rust-1.94.0-nightly" "$stage/lib/libLLVM.so.21.1-rust-1.94.0-nightly"
ln -sfn "$llvm/lib/libLLVM.so.21.1-rust-1.94.0-nightly" "$hostlib/libLLVM-21-rust-1.94.0-nightly.so"
ln -sfn "$installed/bin/cargo" "$stage/bin/cargo"
ln -sfnT "$work/rust-src" "$stage/lib/rustlib/src/rust"
jitdir=$stage/lib/rustlib/x86_64-unknown-linux-gnu/codegen-backends/lib/alphaev6-unknown-linux-gnu
mkdir -p "$jitdir"
ln -sfn "${ALPHA_CROSS:-/home/lindholm/x-tools/alphaev6-unknown-linux-gnu}/lib/libgccjit.so" "$jitdir/libgccjit.so"
if [[ ! -d $work/backend-src ]]; then
    mkdir -p "$work/backend-src"
    rsync -a --exclude=.git --exclude=target --exclude=build "$project/ctng-alpha/rustc_codegen_gcc/" "$work/backend-src/"
    patch -d "$work/backend-src" -p1 < "$repo/patches/rustc_codegen_gcc-symbol-versions.patch"
fi
# This source compiler reports the same version string across local rebuilds.
# Touch the backend root to force relinking against the freshly built driver.
touch "$work/backend-src/src/lib.rs"
export RUSTC=$stage/bin/rustc RUSTC_BOOTSTRAP=1 LD_LIBRARY_PATH=$stage/lib CARGO_TARGET_DIR=$work/backend-target
"$installed/bin/cargo" build --offline --release --manifest-path "$work/backend-src/Cargo.toml"
sha256sum "$stage/bin/rustc" "$stage/lib/"librustc_driver*.so "$CARGO_TARGET_DIR/release/librustc_codegen_gcc.so" | sha256sum | cut -c1-16 > "$work/toolchain-id"
echo "Compiler: $RUSTC"
echo "Backend: $CARGO_TARGET_DIR/release/librustc_codegen_gcc.so"
echo "Build caches are separated by toolchain content hash in scripts/alpha-fixed-env.sh."
