use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub name: String,
    pub values: Vec<u64>,
}

#[repr(C)]
struct Triple { a: u64, b: u64, c: u64 }
unsafe extern "C" {
    fn sdk_native_sum(value: Triple, mask: u32) -> u64;
    fn sdk_native_length(text: *const core::ffi::c_char) -> usize;
}

pub fn native_sum() -> u64 {
    unsafe { sdk_native_sum(Triple { a: 1, b: 1u64 << 40, c: 7 }, 0x80000001) }
}

pub fn validate() {
    assert!(cfg!(target_arch = "alpha"));
    assert_eq!(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }, 8192);
    let record = Record { name: "Alpha EV6".into(), values: vec![1, u64::MAX, 1u64 << 40] };
    let json = serde_json::to_string(&record).unwrap();
    assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), record);
    assert!(regex::Regex::new(r"^Alpha EV\d+$").unwrap().is_match(&record.name));
    assert_eq!(native_sum(), (1u64 << 40) + 0x80000009);
    assert_eq!(unsafe { sdk_native_length(c"hello alpha".as_ptr()) }, 11);
}

#[cfg(test)]
mod tests {
    #[test]
    fn serialization_regex_and_c() { super::validate(); }

    #[test]
    fn threads_and_atomics() {
        use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
        let count = Arc::new(AtomicU64::new(0));
        let threads: Vec<_> = (0..4).map(|_| {
            let count = count.clone();
            std::thread::spawn(move || { for _ in 0..1000 { count.fetch_add(1, Ordering::Relaxed); } })
        }).collect();
        for thread in threads { thread.join().unwrap(); }
        assert_eq!(count.load(Ordering::Relaxed), 4000);
    }

    #[test]
    #[should_panic(expected = "expected Alpha panic")]
    fn expected_panic_is_reported() { panic!("expected Alpha panic"); }

    #[test]
    #[ignore = "runner validation invokes this separately and expects failure"]
    fn intentional_failure() { assert_eq!(1, 2, "failure propagation"); }
}
