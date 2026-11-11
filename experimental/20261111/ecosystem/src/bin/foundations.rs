use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Record {
    name: String,
    values: Vec<i64>,
}
#[derive(Parser)]
struct Options {
    #[arg(long)]
    count: usize,
}
#[derive(Debug, thiserror::Error)]
#[error("invalid Alpha sample: {0}")]
struct SampleError(u32);

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    let record = Record {
        name: "Alpha λ".into(),
        values: vec![i64::MIN, 0, i64::MAX],
    };
    let json = serde_json::to_string(&record)?;
    assert_eq!(serde_json::from_str::<Record>(&json)?, record);
    assert_eq!(
        toml::from_str::<Record>(&toml::to_string(&record)?)?,
        record
    );
    assert!(serde_json::from_str::<Record>("broken").is_err());
    assert_eq!(
        Options::try_parse_from(["probe", "--count", "42"])?.count,
        42
    );
    assert!(Options::try_parse_from(["probe", "--count", "invalid"]).is_err());
    let re = regex::Regex::new(r"(?P<word>\p{Greek}+)=(\d+)")?;
    let captures = re.captures("λ=42").context("Unicode regex")?;
    assert_eq!(&captures["word"], "λ");
    let err = anyhow::Error::new(SampleError(42)).context("outer context");
    assert_eq!(err.downcast_ref::<SampleError>().unwrap().0, 42);
    assert!(format!("{err:#}").contains("invalid Alpha sample: 42"));
    println!("PASS serde/serde_json/toml/clap/regex/anyhow/thiserror");

    let digest = format!("{:x}", Sha256::digest(b"abc"));
    assert_eq!(
        digest,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let a = uuid::Uuid::new_v4();
    let b = uuid::Uuid::new_v4();
    assert_ne!(a, b);
    assert_eq!(a.get_version_num(), 4);
    assert_eq!(uuid::Uuid::parse_str(&a.to_string())?, a);
    use rand::{Rng, SeedableRng};
    let mut seeded = rand::rngs::StdRng::seed_from_u64(42);
    let mut repeated = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..256 {
        assert_eq!(seeded.random::<u64>(), repeated.random::<u64>());
    }
    let mut os_bytes = [0u8; 64];
    rand::rng().fill(&mut os_bytes);
    assert!(os_bytes.iter().any(|b| *b != 0));
    println!("PASS sha2 known vector, rand seeded/OS entropy, uuid v4");

    let data = "Alpha compression λ\n".repeat(8192);
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(data.as_bytes())?;
    let compressed = encoder.finish()?;
    assert!(compressed.len() < data.len());
    let mut decoded = String::new();
    flate2::read::GzDecoder::new(compressed.as_slice()).read_to_string(&mut decoded)?;
    assert_eq!(decoded, data);
    let dir = tempfile::tempdir()?;
    std::fs::create_dir(dir.path().join("nested"))?;
    std::fs::write(dir.path().join("nested/data.gz"), compressed)?;
    let entries: Vec<_> = walkdir::WalkDir::new(dir.path())
        .into_iter()
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(entries.len(), 3);
    let old_path = dir.path().to_owned();
    dir.close()?;
    assert!(!old_path.exists());
    tracing::info!("PASS tracing/tracing-subscriber, flate2, tempfile/walkdir");
    Ok(())
}
