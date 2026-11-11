#![no_std]
#![no_main]
use alpha_libc_probe::*;
use core::{
    ptr,
    sync::atomic::{AtomicBool, Ordering},
};
static SIGNAL: AtomicBool = AtomicBool::new(false);
extern "C" fn handler(signal: libc::c_int) {
    SIGNAL.store(signal == libc::SIGUSR1, Ordering::SeqCst);
}
fn check(ok: bool, label: &[u8]) {
    if !ok {
        print(label);
        print(b": FAIL\n");
        finish(1);
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    unsafe {
        let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
        check(page == 8192, b"page size");
        let memory = libc::mmap(
            ptr::null_mut(),
            page,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        check(memory != libc::MAP_FAILED, b"mmap");
        (memory as *mut u8).write_volatile(17);
        check(
            libc::madvise(memory, page, libc::MADV_DONTNEED) == 0,
            b"madvise",
        );
        check(
            (memory as *mut u8).read_volatile() == 0,
            b"MADV_DONTNEED effect",
        );
        check(
            libc::mprotect(memory, page, libc::PROT_READ) == 0,
            b"mprotect",
        );
        check(libc::munmap(memory, page) == 0, b"munmap");
        print(b"libc mmap, madvise, mprotect: PASS\n");
        let mut pipe = [-1; 2];
        check(
            libc::pipe2(pipe.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) == 0,
            b"pipe2",
        );
        let mut byte = 0u8;
        check(
            libc::read(pipe[0], (&mut byte as *mut u8).cast(), 1) == -1,
            b"nonblocking read",
        );
        check(*libc::__errno_location() == libc::EAGAIN, b"EAGAIN");
        libc::close(pipe[0]);
        libc::close(pipe[1]);
        print(b"libc pipe flags and errno: PASS\n");
        let mut action: libc::sigaction = core::mem::zeroed();
        action.sa_sigaction = handler as *const () as libc::sighandler_t;
        check(libc::sigemptyset(&mut action.sa_mask) == 0, b"sigemptyset");
        let mut previous = core::mem::zeroed();
        check(
            libc::sigaction(libc::SIGUSR1, &action, &mut previous) == 0,
            b"sigaction",
        );
        check(libc::raise(libc::SIGUSR1) == 0, b"raise");
        check(SIGNAL.load(Ordering::SeqCst), b"signal handler");
        check(
            libc::sigaction(libc::SIGUSR1, &previous, ptr::null_mut()) == 0,
            b"restore sigaction",
        );
        print(b"libc signal installation and delivery: PASS\n");
        let mut terminal: libc::termios = core::mem::zeroed();
        check(
            libc::cfsetspeed(&mut terminal, libc::B115200) == 0,
            b"cfsetspeed",
        );
        check(
            libc::cfgetispeed(&terminal) == libc::B115200,
            b"cfgetispeed",
        );
        check(
            libc::cfgetospeed(&terminal) == libc::B115200,
            b"cfgetospeed",
        );
        check(
            libc::cfsetospeed(&mut terminal, libc::B9600) == 0,
            b"cfsetospeed",
        );
        check(libc::cfgetospeed(&terminal) == libc::B9600, b"output baud");
        check(
            terminal.c_cflag & libc::CBAUD == libc::B9600,
            b"encoded baud ABI",
        );
        check(
            libc::cfsetispeed(&mut terminal, libc::B4800) == 0,
            b"cfsetispeed",
        );
        check(libc::cfgetispeed(&terminal) == libc::B4800, b"input baud");
        print(b"libc termios legacy baud ABI: PASS\n");
    }
    finish(0)
}
