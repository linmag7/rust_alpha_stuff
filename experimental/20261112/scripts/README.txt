Run these scripts from the gcc-alpha-rust project root. They are local release
assembly helpers, not a clean-machine compiler bootstrap.

Inputs:
- Original compiler archive and three original prepared application source
  archives under work/portage-20261111/distfiles (also shipped in this kit).
- Latest experimental/20261111/portage source recipes in rust_alpha_stuff.
- inputs/crt.tar in work/release-20261112, copied from native-crt-input.tar here.
- Validated release evidence in work/release-20261112/evidence (copies shipped).

mkdir -p work/release-20261112/{stage,inputs,evidence,dist}
cp rust_alpha_stuff/experimental/20261112/scripts/native-crt-input.tar work/release-20261112/inputs/crt.tar
tar -xJf work/portage-20261111/distfiles/alpha-native-toolchain-20261111-experimental.tar.xz -C work/release-20261112/stage
python3 rust_alpha_stuff/experimental/20261112/scripts/prepare.py
python3 rust_alpha_stuff/experimental/20261112/scripts/make-kit.py
python3 rust_alpha_stuff/experimental/20261112/repack-sdk.py work/release-20261112/stage work/release-20261112/dist/rust-alpha-portage-20261112/distfiles/alpha-native-toolchain-20261112-experimental.tar.xz
cp rust_alpha_stuff/experimental/20261112/evidence/* work/release-20261112/evidence/
python3 rust_alpha_stuff/experimental/20261112/scripts/finish-release.py

Native validation must be rerun if changing payloads, not inferred from copied
logs. validate-sdk.sh takes an isolated native SDK path and test directory.
