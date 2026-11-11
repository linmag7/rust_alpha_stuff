#![no_std]
#![no_main]
use alpha_abi_probe::*;
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe {
        report(b"Testing 16-byte aggregate arguments\n");
        check(
            c_pair_arg(Pair { a: 17, b: 29 }) == 1,
            b"Rust -> C pair argument",
        );
        check(c_call_rust_pair_arg() == 1, b"C -> Rust pair argument");
    }
    finish(0)
}
