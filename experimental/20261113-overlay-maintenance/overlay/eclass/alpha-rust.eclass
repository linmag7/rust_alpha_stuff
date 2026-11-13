# Copyright 2026 Gentoo Authors and private Alpha experiment contributors
# Distributed under the terms of the GNU General Public License v2
# Helpers only: no exported phases. Use alongside cargo.eclass with CARGO_OPTIONAL=1.
case ${EAPI} in 8) ;; *) die "alpha-rust requires EAPI 8" ;; esac
ALPHA_RUST_PATCH_DIR="${BASH_SOURCE[0]%/*}/alpha-rust"
inherit toolchain-funcs
BDEPEND+=" >=dev-lang/rust-alpha-bin-1.102.0_pre20261111:0"

alpha-rust_pkg_setup() {
 [[ ${MERGE_TYPE} == binary ]] && return 0
 case ${CHOST} in alpha*-linux-gnu*) ;; *) die "Native Alpha GNU/Linux only" ;; esac
 tc-is-cross-compiler && die "Native Alpha only"
 [[ -x ${BROOT}/opt/rust-alpha-experimental/bin/cargo-alpha ]] || die "Install rust-alpha-bin"
}

# Call once in src_prepare, before registering any crate overrides.
alpha-rust_prepare() {
 mkdir -p "${T}" || die
 cat > "${T}/alpha-rust-cargo" <<'WRAPPER' || die
#!/bin/bash
set -euo pipefail
: "${ALPHA_RUST_PREFIX:?}"
args=()
WRAPPER
 chmod 0755 "${T}/alpha-rust-cargo" || die
}

# Exact version only; copies and patches privately. nix uses the SDK's alias key.
# Call after prepare, before finish. Multiple sysinfo versions need distinct keys.
alpha-rust_patch_crate() {
 local crate=$1 version=$2 source=$3 key=${4:-$1}
 local patch="${ALPHA_RUST_PATCH_DIR}/${crate}-${version}.patch"
 local dest="${WORKDIR}/alpha-${crate}-${version}"
 [[ -f ${patch} && -d ${source} && ! -e ${dest} ]] || die "Unsupported crate or existing private copy"
 cp -a "${source}" "${dest}" || die
 pushd "${dest}" >/dev/null || die
 eapply "${patch}"
 popd >/dev/null || die
 printf 'args+=( --config %q )\n' "patch.crates-io.${key}.path=\"${dest}\"" >> "${T}/alpha-rust-cargo" || die
}

alpha-rust_finish_prepare() {
 echo 'exec "${ALPHA_RUST_PREFIX}/bin/cargo-alpha" "${args[@]}" "$@"' >> "${T}/alpha-rust-cargo" || die
}

alpha-rust_env() {
	export ALPHA_RUST_PREFIX="${BROOT}/opt/rust-alpha-experimental"
	export CARGO="${T}/alpha-rust-cargo"
	[[ -x ${CARGO} ]] || die "Call alpha-rust_prepare first"
	export RUSTC="${ALPHA_RUST_PREFIX}/bin/rustc"
	export CARGO_HOME="${ECARGO_HOME}" TMPDIR="${T}"
	export CARGO_TARGET_DIR="${WORKDIR}/alpha-target"
	export CARGO_NET_OFFLINE=true CARGO_INCREMENTAL=0
	export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_STRIP=false
	export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
	export CARGO_PROFILE_RELEASE_PANIC=abort
	unset CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS CARGO_BUILD_TARGET
	unset RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
	# The experimental GCC crtbegin lacks __dso_handle. Use only the native
	# Gentoo startup objects, leaving compiler and runtime libraries unchanged.
	local obj crtpath
	mkdir -p "${T}/alpha-crt" || die
	for obj in crtbegin.o crtbeginS.o crtbeginT.o crtend.o crtendS.o; do
		crtpath=$(env -u GCC_EXEC_PREFIX -u COMPILER_PATH -u LIBRARY_PATH \
			"${BROOT}/usr/bin/gcc" -print-file-name="${obj}") || die
		[[ ${crtpath} = /* && -f ${crtpath} ]] || die "Missing native ${obj}"
		cp "${crtpath}" "${T}/alpha-crt/${obj}" || die
	done
	# Alpha needs IEEE-compliant NaN/Inf handling for Rust floating point.
	export RUSTFLAGS="-Cllvm-args=-mieee -Cdebuginfo=0 -Clink-arg=-B${T}/alpha-crt/"
}


alpha-rust_configure() {
 alpha-rust_env
 # Resolve SDK overrides only from the ebuild's downloaded/vendor sources.
 "${CARGO}" generate-lockfile --offline || die
 "${CARGO}" metadata --locked --offline --filter-platform alphaev6-unknown-linux-gnu --format-version 1 "$@" > "${T}/alpha-cargo-metadata.json" || die
 python3 "${ALPHA_RUST_PATCH_DIR}/check-dependencies.py" "${T}/alpha-cargo-metadata.json" "${WORKDIR}" || die "Resolve known Alpha dependency blockers before compiling"
}

alpha-rust_build() {
 alpha-rust_env
 "${CARGO}" build --release --locked --offline -j "$(makeopts_jobs)" "$@" || die
}
