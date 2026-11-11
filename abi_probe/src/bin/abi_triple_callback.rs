#![no_std]
#![no_main]
use alpha_abi_probe::*;
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    report(b"Testing C -> Rust 24-byte aggregate argument\n");
    unsafe {
        check(c_call_rust_triple_arg() == 1, b"C -> Rust triple argument");
    }
    finish(0)
}
