use rayon::prelude::*;
use std::sync::Arc;
fn main() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let sum = pool.install(|| (0..100_000u64).into_par_iter().map(|x| x * x).sum::<u64>());
    assert_eq!(sum, (0..100_000u64).map(|x| x * x).sum::<u64>());
    let (tx, rx) = crossbeam_channel::bounded(17);
    let counter = Arc::new(parking_lot::Mutex::new(0usize));
    std::thread::scope(|scope| {
        for worker in 0..4 {
            let tx = tx.clone();
            let counter = counter.clone();
            scope.spawn(move || {
                for n in 0..1000 {
                    tx.send(worker * 1000 + n).unwrap();
                    *counter.lock() += 1;
                }
            });
        }
        drop(tx);
        let mut received: Vec<_> = rx.iter().collect();
        received.sort_unstable();
        assert_eq!(received, (0..4000).collect::<Vec<_>>());
    });
    assert_eq!(*counter.lock(), 4000);
    let lock = parking_lot::RwLock::new(10);
    *lock.write() += 32;
    assert_eq!(*lock.read(), 42);
    println!("PASS rayon parallel sum, crossbeam bounded channels, parking_lot contended locks");
}
