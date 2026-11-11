#![no_std]

use core::panic::PanicInfo;

unsafe extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn _exit(code: i32) -> !;
    pub fn c_layout() -> u64;
    pub fn c_scalars(u: u32, s: i32, b: u8, h: i16, f: f64, x: u64, y: u64, g: f64) -> u64;
    pub fn c_pair_arg(p: Pair) -> u64;
    pub fn c_triple_arg(p: Triple) -> u64;
    pub fn c_pair_ret() -> Pair;
    pub fn c_call_rust_scalars() -> u64;
    pub fn c_call_rust_pair_arg() -> u64;
    pub fn c_call_rust_triple_arg() -> u64;
    pub fn c_call_rust_pair_ret() -> u64;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pair {
    pub a: u64,
    pub b: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Triple {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}
#[repr(C)]
pub struct Aligned {
    pub tag: u8,
    pub value: u64,
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_scalars(
    u: u32,
    s: i32,
    b: u8,
    h: i16,
    f: f64,
    x: u64,
    y: u64,
    g: f64,
) -> u64 {
    (u == 0xf1234567
        && s == -1234567
        && b == 241
        && h == -1234
        && f == 1.25
        && x == 17
        && y == 29
        && g == -2.5) as u64
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_pair_arg(p: Pair) -> u64 {
    (p.a == 17 && p.b == 29) as u64
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_triple_arg(p: Triple) -> u64 {
    (p.a == 17 && p.b == 29 && p.c == 43) as u64
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_pair_ret() -> Pair {
    Pair { a: 17, b: 29 }
}

pub fn report(message: &[u8]) {
    unsafe {
        write(1, message.as_ptr(), message.len());
    }
}
pub fn check(ok: bool, message: &[u8]) {
    report(message);
    if ok {
        report(b": PASS\n");
    } else {
        report(b": FAIL\n");
        finish(1);
    }
}
pub fn finish(code: i32) -> ! {
    unsafe { _exit(code) }
}
#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    report(b"PANIC\n");
    finish(101)
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Tiny {
    pub value: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Odd {
    pub bytes: [u8; 3],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Large {
    pub words: [u64; 8],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mixed {
    pub real: f64,
    pub word: u32,
}
unsafe extern "C" {
    pub fn c_tiny_ret() -> Tiny;
    pub fn c_odd_ret() -> Odd;
    pub fn c_triple_ret() -> Triple;
    pub fn c_large_arg(p: Large) -> u64;
    pub fn c_split_arg(a: u64, b: u64, c: u64, d: u64, e: u64, p: Triple, tail: u64) -> u64;
    pub fn c_odd_arg(p: Odd) -> u64;
    pub fn c_mixed_arg(p: Mixed) -> u64;
    pub fn c_call_rust_extended() -> u64;
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_tiny_ret() -> Tiny {
    Tiny { value: 0xf1234567 }
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_odd_ret() -> Odd {
    Odd {
        bytes: [17, 29, 43],
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_triple_ret() -> Triple {
    Triple {
        a: 17,
        b: 29,
        c: 43,
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_large_arg(p: Large) -> u64 {
    p.words.iter().enumerate().all(|(i, v)| *v == 17 + i as u64) as u64
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_split_arg(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    p: Triple,
    tail: u64,
) -> u64 {
    (a == 1
        && b == 2
        && c == 3
        && d == 4
        && e == 5
        && p.a == 17
        && p.b == 29
        && p.c == 43
        && tail == 71) as u64
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_odd_arg(p: Odd) -> u64 {
    (p.bytes == [17, 29, 43]) as u64
}
#[unsafe(no_mangle)]
pub extern "C" fn rust_mixed_arg(p: Mixed) -> u64 {
    (p.real == 1.25 && p.word == 0xf1234567) as u64
}
