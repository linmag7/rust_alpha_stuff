#![no_std]
#![no_main]
use alpha_abi_probe::*;
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe {
        report(b"Testing 24-byte aggregate arguments\n");
        check(
            c_triple_arg(Triple {
                a: 17,
                b: 29,
                c: 43,
            }) == 1,
            b"Rust -> C triple argument",
        );
        check(c_call_rust_triple_arg() == 1, b"C -> Rust triple argument");
    }
    finish(0)
}
