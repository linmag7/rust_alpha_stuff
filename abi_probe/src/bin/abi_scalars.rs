#![no_std]
#![no_main]
use alpha_abi_probe::*;
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe {
        check(
            c_layout() == 1
                && core::mem::size_of::<Aligned>() == 16
                && core::mem::offset_of!(Aligned, value) == 8
                && core::mem::align_of::<u64>() == 8,
            b"C/Rust layout",
        );
        check(
            c_scalars(0xf1234567, -1234567, 241, -1234, 1.25, 17, 29, -2.5) == 1,
            b"Rust -> C scalars and stack arguments",
        );
        check(
            c_call_rust_scalars() == 1,
            b"C -> Rust scalars and stack arguments",
        );
    }
    finish(0)
}
