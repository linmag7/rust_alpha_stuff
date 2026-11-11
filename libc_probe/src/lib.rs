#![no_std]
use core::panic::PanicInfo;
pub fn print(bytes: &[u8]) {
    unsafe { libc::write(1, bytes.as_ptr().cast(), bytes.len()); }
}
pub fn number(n:u64) {
    let mut bytes=[b'0';16];
    for (i,b) in bytes.iter_mut().enumerate() {
        let nibble=((n >> (60-i*4))&15) as u8;
        *b=if nibble<10 {b'0'+nibble} else {b'a'+nibble-10};
    }
    print(&bytes);
}
pub fn finish(status:i32)->! {unsafe{libc::_exit(status)}}
#[panic_handler]
fn panic(_: &PanicInfo)->! {print(b"PANIC\n");finish(101)}
