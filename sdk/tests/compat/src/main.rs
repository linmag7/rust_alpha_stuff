use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "Alpha")]
    name: String,
    #[arg(long, default_value_t = 0)]
    exit_code: u8,
    #[arg(long)]
    read_stdin: bool,
    #[arg(long)]
    input: Option<std::path::PathBuf>,
}

fn main() {
    let args = Args::parse();
    alpha_sdk_compat::validate();
    println!("SDK compatibility: PASS; hello {}", args.name);
    if args.read_stdin {
        use std::io::Read;
        let mut value = String::new();
        std::io::stdin().read_to_string(&mut value).unwrap();
        println!("stdin: {}", value.trim_end());
    }
    if let Some(path) = args.input { println!("fixture: {}", std::fs::read_to_string(path).unwrap().trim_end()); }
    if let Ok(value) = std::env::var("ALPHA_SDK_TEST_VALUE") { eprintln!("environment: {value}"); }
    if args.exit_code != 0 { std::process::exit(args.exit_code.into()); }
}
