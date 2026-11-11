use std::{env, path::PathBuf, process::Command};

fn main() {
    assert_eq!(env::var("CARGO_CFG_TARGET_ARCH").unwrap(), "alpha");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("abi.o");
    let archive = out.join("libalpha_abi.a");
    assert!(
        Command::new(env::var("ALPHA_CC").expect("use scripts/alpha-build.sh"))
            .arg(format!("--sysroot={}", env::var("ALPHA_SYSROOT").unwrap()))
            .args([
                "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-c", "abi.c", "-o"
            ])
            .arg(&object)
            .status()
            .unwrap()
            .success()
    );
    assert!(
        Command::new(env::var("ALPHA_AR").unwrap())
            .arg("crs")
            .arg(&archive)
            .arg(&object)
            .status()
            .unwrap()
            .success()
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=alpha_abi");
    // The saved Alpha binutils produces DT_JMPREL=0 in these small PIEs.
    // Keep this workaround local to ABI probes, and allow reproducing the PIE bug.
    if env::var_os("ALPHA_PROBE_PIE").is_none() {
        println!("cargo:rustc-link-arg=-no-pie");
    }
    println!("cargo:rerun-if-env-changed=ALPHA_PROBE_PIE");
    println!("cargo:rerun-if-changed=abi.c");
    for name in ["ALPHA_CC", "ALPHA_AR", "ALPHA_SYSROOT"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
}
