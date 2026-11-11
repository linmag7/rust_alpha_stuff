fn main() {
    cc::Build::new().file("native.c").warnings_into_errors(true).compile("alpha_sdk_native");
    println!("cargo:rerun-if-changed=native.c");
}
