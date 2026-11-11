use anyhow::Result;
use axum::{Json, Router, routing::get};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), exercise()).await??;
    Ok(())
}
async fn exercise() -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    let producer = tokio::spawn(async move {
        for i in 0..100 {
            tx.send(i).await.unwrap();
        }
    });
    let mut sum = 0;
    while let Some(i) = rx.recv().await {
        sum += i;
    }
    producer.await?;
    assert_eq!(sum, 4950);
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(tokio::task::spawn_blocking(|| 6 * 7).await?, 42);
    let (mut reader, mut writer) = tokio::io::duplex(8);
    let sender = tokio::spawn(async move {
        writer.write_all(b"alpha async IO").await.unwrap();
    });
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await?;
    sender.await?;
    assert_eq!(bytes, b"alpha async IO");
    println!("PASS tokio multithread tasks, timers, channels, blocking pool, async IO");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let app = Router::new().route(
        "/alpha",
        get(|| async { Json(serde_json::json!({"arch": "alpha", "answer": 42})) }),
    );
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
    });
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()?;
    for _ in 0..4 {
        let response = client
            .get(format!("http://{address}/alpha"))
            .send()
            .await?
            .error_for_status()?;
        assert_eq!(
            response.json::<serde_json::Value>().await?,
            serde_json::json!({"arch": "alpha", "answer": 42})
        );
    }
    assert_eq!(
        client
            .get(format!("http://{address}/missing"))
            .send()
            .await?
            .status(),
        404
    );
    drop(client);
    shutdown_tx.send(()).unwrap();
    server.await??;
    let udp_a = tokio::net::UdpSocket::bind("127.0.0.1:0").await?;
    let udp_b = tokio::net::UdpSocket::bind("127.0.0.1:0").await?;
    udp_a.send_to(b"alpha udp", udp_b.local_addr()?).await?;
    let mut payload = [0u8; 32];
    let (len, _) = udp_b.recv_from(&mut payload).await?;
    assert_eq!(&payload[..len], b"alpha udp");
    println!("PASS axum/reqwest HTTP+JSON over loopback TCP, 404, graceful shutdown, tokio UDP");
    Ok(())
}
