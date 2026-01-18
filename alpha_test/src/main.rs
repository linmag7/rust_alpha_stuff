#![no_std]
#![no_main]

#[repr(C)]
#[derive(Copy, Clone)]
struct PairU64 {
    a: u64,
    b: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct QuadU32 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

fn print_pair(tag: &[u8], p: PairU64) {
    print(tag);
    print(b"{a="); print_u64_hex(p.a); print(b", b="); print_u64_hex(p.b); print(b"}\n");
}


extern crate alloc;

use core::panic::PanicInfo;
use alloc::boxed::Box;
use alloc::vec::Vec;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    print(b"PANIC\n");
    unsafe { _exit(111) }
}

#[inline(never)]
fn abi_stress_callee(
    p: PairU64,          // small aggregate by-value
    q: QuadU32,          // another aggregate by-value
    x0: u64,
    x1: u64,
    x2: u64,
    x3: u64,
    x4: u64,
    x5: u64,             // enough ints to likely spill to stack
    s0: i64,
    u0: u32,
    u1: u32,
    f0: f64,
    f1: f64,
    ptr: *const u8,      // pointer arg
) -> PairU64 {           // small aggregate return
    // Mix everything so optimizer can't trivially drop it.
    let mut h0 = p.a ^ x0 ^ x2 ^ x4 ^ (s0 as u64) ^ (u0 as u64) ^ ((q.a as u64) << 32 | q.b as u64);
    let mut h1 = p.b ^ x1 ^ x3 ^ x5 ^ (u1 as u64) ^ ((q.c as u64) << 32 | q.d as u64);

    // Incorporate floats in a reproducible way
    let fb0 = f0.to_bits();
    let fb1 = f1.to_bits();
    h0 ^= fb0.rotate_left(7);
    h1 ^= fb1.rotate_right(11);

    // Touch memory through ptr (but safely) to exercise pointer passing
    let m = unsafe { *ptr } as u64;
    h0 = h0.wrapping_add(m);
    h1 = h1.wrapping_mul(3).wrapping_add(m ^ 0x55);

    PairU64 { a: h0, b: h1 }
}


unsafe extern "C" {
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn _exit(code: i32) -> !;
}

const PAGE: usize = 4096;

fn round_up_page(x: usize) -> usize {
    (x + PAGE - 1) & !(PAGE - 1)
}

/// Print raw bytes to stdout (fd = 1)
#[inline(always)]
fn print(msg: &[u8]) {
    unsafe {
        let _ = write(1, msg.as_ptr(), msg.len());
    }
}
use core::alloc::{GlobalAlloc, Layout};

use core::ffi::c_void;
//use libc::{PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANONYMOUS, MAP_FAILED};

const PROT_READ: i32  = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 2;
//const MAP_ANONYMOUS: i32 = 0x20;

// Alpha/Linux likely uses 0x1000 for MAP_ANONYMOUS (see below)
const MAP_ANONYMOUS: i32 = 0x10;

const MAP_FAILED: *mut c_void = (-1isize) as *mut c_void;
//use core::ffi::c_void;

type OffT = i64;


unsafe extern "C" {
    fn __errno_location() -> *mut i32;
}

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: OffT,
    ) -> *mut c_void;

    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

struct MmapAlloc;


unsafe impl GlobalAlloc for MmapAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        print(b"alloc: size="); print_i64_dec(layout.size() as i64); print_nl();
        let size = round_up_page(layout.size().max(PAGE));

        let raw = unsafe { mmap(
            core::ptr::null_mut(),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0 as OffT,
        ) };

        print(b"mmap -> ");
        print_u64_hex(raw as u64);
        print_nl();
        if raw == MAP_FAILED {
    let e = unsafe { *__errno_location() };
    print(b"mmap failed errno="); print_i64_dec(e as i64); print_nl();
    return core::ptr::null_mut();
}
        if raw == MAP_FAILED {
            core::ptr::null_mut()
        } else {
            raw as *mut u8
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = round_up_page(layout.size().max(PAGE));
        print(b"dealloc: size="); print_i64_dec(layout.size() as i64); print_nl();
        print(b"dealloc: ptr -> ");
        print_u64_hex(ptr as u64);
        print_nl();
        let _ = unsafe { munmap(ptr as *mut c_void, size) };
    }
}

#[global_allocator]
static ALLOC: MmapAlloc = MmapAlloc;

/* --------------------------------------------------------- */
/* Minimal helpers (no formatting infrastructure available)  */
/* --------------------------------------------------------- */

fn print_u64_hex(v: u64) {
    let mut buf = [b'0'; 18];
    buf[0] = b'0';
    buf[1] = b'x';

    for i in 0..16 {
        let shift = 60 - i * 4;
        let d = ((v >> shift) & 0xf) as u8;
        buf[2 + i] = if d < 10 { b'0' + d } else { b'a' + (d - 10) };
    }

    print(&buf);
}

fn print_u64_dec(mut v: u64) {
    let mut buf = [0u8; 32];
    let mut i = 0;

    if v == 0 {
        print(b"0");
        return;
    }

    while v > 0 {
        buf[i] = b'0' + (v % 10) as u8;
        i += 1;
        v /= 10;
    }

    while i > 0 {
        i -= 1;
        print(&buf[i..i + 1]);
    }
}

fn print_i64_dec(v: i64) {
    if v < 0 {
        print(b"-");
        // avoid overflow on i64::MIN
        print_u64_dec(v.wrapping_neg() as u64);
    } else {
        print_u64_dec(v as u64);
    }
}

fn print_nl() {
    print(b"\n");
}

/* --------------------------------------------------------- */
/* Integer math test                                         */
/* --------------------------------------------------------- */

fn do_some_integer_math() {
    print(b"\n== integer math ==\n");
    print(b"dec test 4096 = "); print_u64_dec(4096); print_nl();
    print(b"dec test 1234567890 = "); print_u64_dec(1_234_567_890); print_nl();

    let a: i64 = 0x1234_5678_9abc_def0;
    let b: i64 = -0x1111_2222_3333_4444;

    let sum = a.wrapping_add(b);
    let diff = a.wrapping_sub(b);
    let prod = a.wrapping_mul(7);
    let quo = a / 3;
    let rem = a % 3;

    print(b"a      = "); print_u64_hex(a as u64); print_nl();
    print(b"b      = "); print_u64_hex(b as u64); print_nl();
    print(b"a+b    = "); print_u64_hex(sum as u64); print_nl();
    print(b"a-b    = "); print_u64_hex(diff as u64); print_nl();
    print(b"a*7    = "); print_u64_hex(prod as u64); print_nl();

    print(b"a/3    = "); print_u64_dec(quo as u64);
    print(b"  rem "); print_u64_dec(rem as u64); print_nl();
    print(b"a/3 (hex) = "); print_u64_hex(quo as u64); print_nl();
    print(b"a%3 (hex) = "); print_u64_hex(rem as u64); print_nl();
    
    let recomposed = quo.wrapping_mul(3).wrapping_add(rem);
    print(b"recomposed = "); print_u64_hex(recomposed as u64); print_nl();
    print(b"orig       = "); print_u64_hex(a as u64); print_nl();
    
    let u: u64 = 0xfedc_ba98_7654_3210;
    let v: u64 = 0x0123_4567_89ab_cdef;

    print(b"u&v    = "); print_u64_hex(u & v); print_nl();
    print(b"u|v    = "); print_u64_hex(u | v); print_nl();
    print(b"u^v    = "); print_u64_hex(u ^ v); print_nl();
    print(b"u<<13  = "); print_u64_hex(u.wrapping_shl(13)); print_nl();
    print(b"u>>17  = "); print_u64_hex(u >> 17); print_nl();
}

/* --------------------------------------------------------- */
/* Floating-point math test (f64)                             */
/* --------------------------------------------------------- */

fn do_some_float_math() {
    print(b"\n== float math (f64) ==\n");

    let x: f64 = 1.0 / 10.0;
    let y: f64 = 3.0;

    let a: f64 = x * y;
    let b: f64 = x + x + x;

    let z: f64 = 1.0e150;
    let c: f64 = (z * z) / z;

    let d: f64 = (1.0e16 + 1.0) - 1.0e16;

    print(b"x*y bits        = "); print_u64_hex(a.to_bits()); print_nl();
    print(b"x+x+x bits      = "); print_u64_hex(b.to_bits()); print_nl();
    print(b"(z*z)/z bits     = "); print_u64_hex(c.to_bits()); print_nl();
    print(b"(1e16+1)-1e16    = "); print_u64_hex(d.to_bits()); print_nl();

    print(b"z bits          = "); print_u64_hex(z.to_bits()); print_nl();

    let eq = if c.to_bits() == z.to_bits() { 1u8 } else { 0u8 };
    print(b"(z*z)/z == z ?  "); print(&[b'0' + eq]); print_nl();

    let pz: f64 = 0.0;
    let nz: f64 = -0.0;
    print(b"+0 bits         = "); print_u64_hex(pz.to_bits()); print_nl();
    print(b"-0 bits         = "); print_u64_hex(nz.to_bits()); print_nl();

}

fn abi_stress_test() {
    print(b"\n== ABI stress ==\n");

    let p = PairU64 { a: 0x1111_2222_3333_4444, b: 0xaaaa_bbbb_cccc_dddd };
    let q = QuadU32 { a: 1, b: 2, c: 3, d: 4 };

    let x0 = 0x0123_4567_89ab_cdef;
    let x1 = 0xfedc_ba98_7654_3210;
    let x2 = 0x0f0e_0d0c_0b0a_0908;
    let x3 = 0x8070_6050_4030_2010;
    let x4 = 0x1357_9bdf_2468_ace0;
    let x5 = 0xdead_beef_cafe_f00d;

    let s0: i64 = -0x1234_5678_9abc_def; // negative to test sign extension
    let u0: u32 = 0x89ab_cdef;
    let u1: u32 = 0x0123_4567;

    let f0: f64 = 1.0 / 10.0;
    let f1: f64 = 3.141592653589793;

    let mem = [0x5au8];
    let ptr = mem.as_ptr();

    let got = abi_stress_callee(p, q, x0, x1, x2, x3, x4, x5, s0, u0, u1, f0, f1, ptr);

    // Expected recomputation (must match callee)
    let mut h0 = p.a ^ x0 ^ x2 ^ x4 ^ (s0 as u64) ^ (u0 as u64) ^ ((q.a as u64) << 32 | q.b as u64);
    let mut h1 = p.b ^ x1 ^ x3 ^ x5 ^ (u1 as u64) ^ ((q.c as u64) << 32 | q.d as u64);

    let fb0 = f0.to_bits();
    let fb1 = f1.to_bits();
    h0 ^= fb0.rotate_left(7);
    h1 ^= fb1.rotate_right(11);

    let m = mem[0] as u64;
    h0 = h0.wrapping_add(m);
    h1 = h1.wrapping_mul(3).wrapping_add(m ^ 0x55);

    let exp = PairU64 { a: h0, b: h1 };

    print_pair(b"got = ", got);
    print_pair(b"exp = ", exp);

    let ok = (got.a == exp.a) && (got.b == exp.b);
    print(b"abi ok? "); print(&[if ok { b'1' } else { b'0' }]); print_nl();
}

/* --------------------------------------------------------- */
/* Entry point                                               */
/* --------------------------------------------------------- */

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    print(b"alpha no_std rust test\n");

    do_some_integer_math();
    do_some_float_math();
    let b = Box::new(0x12345678_u64);
    print(b"Box value = ");
    print_u64_hex(*b);
    print_nl();


    let mut v = Vec::new();
    for i in 0..16 {
        v.push(i * 3);
    }
    
    abi_stress_test();

    unsafe {
        _exit(0);
    }
}

