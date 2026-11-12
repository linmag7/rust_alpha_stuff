#!/bin/bash
set -euo pipefail
[[ $(hostname) == ansgar && $(uname -m) == alpha ]] || exit 90
base=/root/rust-alpha-overnight-20261111/crt-probe
mkdir -p "$base/crt"
cd "$base"
for obj in crtbegin.o crtbeginS.o crtbeginT.o crtend.o crtendS.o; do
 path=$(env -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH /usr/bin/gcc -print-file-name="$obj")
 [[ $path = /* && -f $path ]] || exit 1
 cp "$path" crt/
done
cat > atfork.c <<'C'
#include <pthread.h>
int main(void) { return pthread_atfork(0,0,0); }
C
if /opt/rust-alpha-experimental/bin/alpha-cc -fno-lto -no-pie atfork.c -o broken > original.log 2>&1; then
 echo 'Unexpected original link success'; exit 1
fi
grep __dso_handle original.log
/opt/rust-alpha-experimental/bin/alpha-cc -B"$base/crt/" -fno-lto -no-pie atfork.c -o fixed
./fixed
cat > tls.rs <<'RS'
use std::sync::atomic::{AtomicUsize, Ordering};
static DROPS: AtomicUsize = AtomicUsize::new(0);
struct Guard;
impl Drop for Guard { fn drop(&mut self) { DROPS.fetch_add(1, Ordering::SeqCst); } }
thread_local! { static GUARD: Guard = const { Guard }; }
fn main() {
 std::thread::spawn(|| GUARD.with(|_| ())).join().unwrap();
 assert_eq!(DROPS.load(Ordering::SeqCst), 1);
 println!("TLS destructor PASS");
}
RS
export LD_LIBRARY_PATH=/opt/rust-alpha-experimental/lib
export CG_GCCJIT_LIBRARY_PATH=/opt/rust-alpha-experimental/lib/libgccjit.so
/opt/rust-alpha-experimental/bin/rustc tls.rs -Cpanic=abort -Clto=off -Clinker=/opt/rust-alpha-experimental/bin/alpha-cc -Clink-arg=-B"$base/crt/" -o tls
./tls
sha256sum crt/*
