#[test]
fn public_api_and_native_abi() {
    alpha_sdk_compat::validate();
}

#[test]
fn files_and_tcp() {
    use std::io::{Read, Write};
    let path = std::env::temp_dir().join(format!("alpha-sdk-test-{}", std::process::id()));
    std::fs::write(&path, b"Alpha file test").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"Alpha file test");
    std::fs::remove_file(path).unwrap();
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let worker = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.write_all(b"alpha").unwrap();
    });
    let mut received = String::new();
    std::net::TcpStream::connect(address).unwrap().read_to_string(&mut received).unwrap();
    worker.join().unwrap();
    assert_eq!(received, "alpha");
}
