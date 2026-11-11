#!/bin/bash
set -euo pipefail
base=/root/rust-alpha-install-20261111
prefix=/opt/rust-alpha-experimental
mkdir -p "$base/native-smoke/src" "$base/native-smoke/tmp" "$base/native-smoke/cargo-home"
cd "$base/native-smoke"
export TMPDIR="$PWD/tmp" CARGO_HOME="$PWD/cargo-home" CARGO_TARGET_DIR="$PWD/target"
cat > Cargo.toml <<'EOF'
[package]
name = "es40-native-smoke"
version = "0.1.0"
edition = "2024"
[workspace]
EOF
cat > src/main.rs <<'EOF'
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
fn main() {
    let counter = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..4).map(|_| {
        let counter = counter.clone();
        std::thread::spawn(move || { for _ in 0..1000 { counter.fetch_add(1, Ordering::SeqCst); } })
    }).collect();
    for handle in handles { handle.join().unwrap(); }
    assert_eq!(counter.load(Ordering::SeqCst), 4000);
    let path = std::env::temp_dir().join(format!("es40-native-smoke-{}", std::process::id()));
    std::fs::write(&path, "Alpha λ\n").unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "Alpha λ\n");
    std::fs::remove_file(path).unwrap();
    println!("ES40 native Rust: threads, atomics and file I/O PASS");
}
EOF
cat > native-c.c <<'EOF'
#include <stdio.h>
int main(void) { return puts("ES40 native GCC C PASS") < 0; }
EOF
cat > native-cxx.cc <<'EOF'
#include <iostream>
#include <stdexcept>
int main() {
    try { throw std::runtime_error("ES40 native G++ exceptions PASS"); }
    catch (const std::runtime_error &e) { std::cout << e.what() << '\n'; return 0; }
    return 1;
}
EOF
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$prefix/bin/alpha-cc" native-c.c -o native-c
./native-c
"$prefix/bin/alpha-c++" native-cxx.cc -o native-cxx
./native-cxx
"$prefix/bin/rustc" -Vv
"$prefix/bin/cargo-alpha" run --offline
for spec in 'sys-apps/ripgrep rg' 'sys-apps/fd fd' 'app-shells/zoxide zoxide'; do
 read -r pkg exe <<< "$spec"
 /usr/bin/"$exe" --version
 bash "/var/db/repos/rust-alpha/$pkg/files/smoke.sh" "/usr/bin/$exe" "$base/installed-smoke/$exe"
done
portageq best_version / dev-lang/rust-alpha-bin
portageq best_version / sys-apps/ripgrep
portageq best_version / sys-apps/fd
portageq best_version / app-shells/zoxide
