#!/bin/bash
set -euo pipefail
[[ $(uname -m) == alpha ]] || exit 90
prefix=${1:?isolated native SDK path required}
base=${2:?test workspace required}
mkdir -p "$base/src" "$base/cargo-home"
cat > "$base/Cargo.toml" <<'TOML'
[package]
name="alpha_release_check"
version="0.1.0"
edition="2021"
TOML
cat > "$base/src/main.rs" <<'RS'
use std::{hint::black_box, sync::atomic::{AtomicUsize,Ordering}};
static DROPS: AtomicUsize=AtomicUsize::new(0);
struct D;
impl Drop for D {fn drop(&mut self){DROPS.fetch_add(1,Ordering::SeqCst);}}
thread_local!{static ITEM: D=D;}
fn main(){
 let n=black_box(f64::NAN);assert!(!(n==black_box(1.0)));assert!(!(n<black_box(1.0)));
 assert!((black_box(1.0f64)/black_box(0.0)).is_infinite());
 std::thread::spawn(||ITEM.with(|_|())).join().unwrap();
 assert_eq!(DROPS.load(Ordering::SeqCst),1);
 std::fs::write("probe.txt",b"alpha").unwrap();assert_eq!(std::fs::read("probe.txt").unwrap(),b"alpha");
 println!("IEEE arithmetic, threads, TLS destructor and file I/O: PASS");
}
RS
cat > "$base/atfork.c" <<'C'
#include <pthread.h>
int main(void) { return pthread_atfork(0,0,0); }
C
cd "$base"
"$prefix/bin/alpha-cc" atfork.c -pthread -o atfork
./atfork
export CARGO_HOME="$base/cargo-home" CARGO_TARGET_DIR="$base/target"
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
"$prefix/bin/cargo-alpha" run --offline --manifest-path "$base/Cargo.toml"
"$prefix/bin/cargo-alpha" run --release --offline --manifest-path "$base/Cargo.toml"
echo 'Native SDK defaults and C pthread_atfork link/run: PASS'
