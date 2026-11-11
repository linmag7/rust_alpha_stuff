use anyhow::Result;
use rusqlite::{Connection, params};
fn main() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("alpha.sqlite");
    let mut db = Connection::open(&path)?;
    db.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE samples (id INTEGER PRIMARY KEY, label TEXT, bytes BLOB, value REAL)")?;
    let tx = db.transaction()?;
    for id in 0..128i64 {
        tx.execute(
            "INSERT INTO samples VALUES (?1, ?2, ?3, ?4)",
            params![id, "Alpha λ", &id.to_le_bytes()[..], id as f64 + 0.25],
        )?;
    }
    tx.commit()?;
    let tx = db.transaction()?;
    tx.execute("DELETE FROM samples", [])?;
    tx.rollback()?;
    drop(db);
    let db = Connection::open(&path)?;
    let count: i64 = db.query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))?;
    assert_eq!(count, 128);
    let (label, bytes, value): (String, Vec<u8>, f64) = db.query_row(
        "SELECT label, bytes, value FROM samples WHERE id=42",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    assert_eq!(label, "Alpha λ");
    assert_eq!(bytes, 42i64.to_le_bytes());
    assert_eq!(value, 42.25);
    let integrity: String = db.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    assert_eq!(integrity, "ok");
    println!(
        "PASS rusqlite/bundled SQLite {}, WAL, commit/rollback, persistent rows, text/blob/float FFI, integrity",
        rusqlite::version()
    );
    Ok(())
}
