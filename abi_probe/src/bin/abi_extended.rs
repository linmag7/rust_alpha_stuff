#![no_std]
#![no_main]
use alpha_abi_probe::*;
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe {
        report(b"Testing aggregate sizes, returns and register/stack split\n");
        check(c_tiny_ret().value == 0xf1234567, b"Rust -> C 4-byte return");
        check(
            c_odd_ret().bytes == [17, 29, 43],
            b"Rust -> C 3-byte return",
        );
        let p = c_triple_ret();
        check(
            p.a == 17 && p.b == 29 && p.c == 43,
            b"Rust -> C 24-byte return",
        );
        check(
            c_large_arg(Large {
                words: [17, 18, 19, 20, 21, 22, 23, 24],
            }) == 1,
            b"Rust -> C 64-byte argument",
        );
        check(
            c_split_arg(1, 2, 3, 4, 5, p, 71) == 1,
            b"Rust -> C split register/stack argument",
        );
        check(
            c_odd_arg(Odd {
                bytes: [17, 29, 43],
            }) == 1,
            b"Rust -> C 3-byte argument",
        );
        check(
            c_mixed_arg(Mixed {
                real: 1.25,
                word: 0xf1234567,
            }) == 1,
            b"Rust -> C float/integer aggregate",
        );
        check(
            c_call_rust_extended() == 0,
            b"C -> Rust extended aggregates",
        );
    }
    finish(0)
}
