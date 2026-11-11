#![no_std]
#![no_main]
use alpha_abi_probe::*;
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe {
        report(b"Testing 16-byte aggregate returns\n");
        let p = c_pair_ret();
        check(p.a == 17 && p.b == 29, b"Rust -> C pair return");
        check(c_call_rust_pair_ret() == 1, b"C -> Rust pair return");
    }
    finish(0)
}
