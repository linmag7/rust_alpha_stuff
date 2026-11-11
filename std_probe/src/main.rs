use std::{
    cell::Cell,
    fs,
    io::{ErrorKind, Read, Write},
    net::{TcpListener, TcpStream, UdpSocket},
    os::unix::fs::{MetadataExt, PermissionsExt},
    sync::{
        Arc, Barrier, Condvar, Mutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
thread_local! { static TLS: Cell<u64> = const { Cell::new(7) }; }
fn main() {
    println!("Rust std on Alpha Linux");
    let values: Vec<u64> = (0..16).map(|i| i * 3).collect();
    assert_eq!(values.iter().sum::<u64>(), 360);
    let text = format!("{} {:.2}", -1234567, 1.25);
    assert_eq!(text, "-1234567 1.25");
    let map: std::collections::HashMap<_, _> = (0..128).map(|n| (n, n * 3)).collect();
    assert_eq!(map.get(&71), Some(&213));
    println!("allocation, formatting and randomized hashing: PASS");

    // Keep test files alongside the executable, on the selected project disk.
    let directory = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .join(format!("std-probe-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let file = directory.join("data");
    fs::write(&file, b"Alpha std filesystem\n").unwrap();
    fs::set_permissions(&file, fs::Permissions::from_mode(0o640)).unwrap();
    let metadata = fs::metadata(&file).unwrap();
    assert_eq!(metadata.len(), 21);
    assert_eq!(metadata.mode() & 0o777, 0o640);
    assert!(metadata.blksize() > 0);
    assert!(metadata.modified().unwrap() <= SystemTime::now());
    assert_eq!(fs::read(&file).unwrap(), b"Alpha std filesystem\n");
    assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
    fs::remove_dir_all(&directory).unwrap();
    println!("files, stat, directory entries and timestamps: PASS");

    let counter = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(3));
    let mut threads = Vec::new();
    for id in 0..2 {
        let counter = counter.clone();
        let barrier = barrier.clone();
        threads.push(
            thread::Builder::new()
                .name(format!("alpha-{id}"))
                .spawn(move || {
                    TLS.with(|v| {
                        assert_eq!(v.get(), 7);
                        v.set(100 + id);
                    });
                    barrier.wait();
                    for _ in 0..10000 {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                    TLS.with(|v| assert_eq!(v.get(), 100 + id));
                })
                .unwrap(),
        );
    }
    barrier.wait();
    for handle in threads {
        handle.join().unwrap();
    }
    assert_eq!(counter.load(Ordering::SeqCst), 20000);
    TLS.with(|v| assert_eq!(v.get(), 7));
    let rw = RwLock::new(17);
    *rw.write().unwrap() += 12;
    assert_eq!(*rw.read().unwrap(), 29);
    let pair = Arc::new((Mutex::new(false), Condvar::new()));
    let other = pair.clone();
    let handle = thread::spawn(move || {
        let (lock, cond) = &*other;
        *lock.lock().unwrap() = true;
        cond.notify_one();
    });
    let (lock, cond) = &*pair;
    let (ready, result) = cond
        .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(5), |v| !*v)
        .unwrap();
    assert!(*ready && !result.timed_out());
    handle.join().unwrap();
    println!("threads, TLS, atomics, locks and condition variables: PASS");

    let start = Instant::now();
    thread::sleep(Duration::from_millis(20));
    assert!(start.elapsed() >= Duration::from_millis(20));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    assert_eq!(listener.accept().unwrap_err().kind(), ErrorKind::WouldBlock);
    listener.set_nonblocking(false).unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut data = [0u8; 4];
        stream.read_exact(&mut data).unwrap();
        assert_eq!(&data, b"ping");
        stream.write_all(b"pong").unwrap();
    });
    let mut client = TcpStream::connect_timeout(&address, Duration::from_secs(5)).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    client.write_all(b"ping").unwrap();
    let mut data = [0u8; 4];
    client.read_exact(&mut data).unwrap();
    assert_eq!(&data, b"pong");
    server.join().unwrap();
    let udp = UdpSocket::bind("127.0.0.1:0").unwrap();
    udp.set_nonblocking(true).unwrap();
    assert_eq!(
        udp.recv_from(&mut data).unwrap_err().kind(),
        ErrorKind::WouldBlock
    );
    println!("clocks, sleep, TCP, UDP and nonblocking errno: PASS");
    let child = std::process::Command::new("/bin/sh")
        .args(["-c", "printf alpha-child; exit 7"])
        .output()
        .unwrap();
    assert_eq!(child.status.code(), Some(7));
    assert_eq!(child.stdout, b"alpha-child");
    assert!(child.stderr.is_empty());
    println!("process creation, pipes and exit status: PASS");
    println!("std probe: PASS");
}
