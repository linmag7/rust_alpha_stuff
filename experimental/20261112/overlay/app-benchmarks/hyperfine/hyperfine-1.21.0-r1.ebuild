# Copyright 2020-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

RUST_MIN_VER="1.97.0"

CRATES="
	aho-corasick@1.1.5
	anstream@1.0.0
	anstyle-parse@1.0.0
	anstyle-query@1.1.5
	anstyle-wincon@3.0.11
	anstyle@1.0.14
	anyhow@1.0.104
	approx@0.5.1
	arrayvec@0.7.8
	assert_cmd@2.2.2
	autocfg@1.5.1
	bitflags@2.13.2
	borsh-derive@1.8.1
	borsh@1.8.1
	bstr@1.13.1
	bumpalo@3.20.3
	bytes@1.12.1
	cfg-if@1.0.5
	cfg_aliases@0.2.2
	chacha20@0.10.2
	clap@4.6.7
	clap_builder@4.6.7
	clap_complete@4.6.11
	clap_lex@1.1.1
	colorchoice@1.0.5
	colored@3.1.1
	console@0.15.11
	console@0.16.6
	cpufeatures@0.3.1
	csv-core@0.1.13
	csv@1.4.0
	difflib@0.4.0
	encode_unicode@1.0.0
	equivalent@1.0.2
	errno@0.3.14
	fastrand@2.5.0
	float-cmp@0.10.0
	getrandom@0.2.17
	getrandom@0.3.4
	getrandom@0.4.3
	hashbrown@0.17.1
	indexmap@2.14.2
	indicatif@0.17.4
	insta-cmd@0.7.0
	insta@1.48.0
	instant@0.1.13
	is_terminal_polyfill@1.70.2
	itoa@1.0.18
	libc@0.2.189
	linux-raw-sys@0.12.1
	memchr@2.8.3
	nix@0.31.3
	normalize-line-endings@0.3.0
	num-traits@0.2.19
	number_prefix@0.4.0
	once_cell@1.21.4
	once_cell_polyfill@1.70.2
	portable-atomic@1.15.0
	ppv-lite86@0.2.21
	predicates-core@1.0.10
	predicates-tree@1.0.13
	predicates@3.1.4
	proc-macro-crate@3.5.0
	proc-macro2@1.0.107
	quote@1.0.47
	r-efi@5.3.0
	r-efi@6.0.0
	rand@0.10.3
	rand@0.8.8
	rand@0.9.5
	rand_chacha@0.3.1
	rand_chacha@0.9.0
	rand_core@0.10.1
	rand_core@0.6.4
	rand_core@0.9.5
	regex-automata@0.4.18
	regex-syntax@0.8.11
	regex@1.13.1
	rust_decimal@1.43.0
	rustix@1.1.5
	rustversion@1.0.23
	ryu@1.0.23
	serde@1.0.229
	serde_core@1.0.229
	serde_derive@1.0.229
	serde_json@1.0.151
	shell-words@1.1.1
	similar@2.7.0
	strip-ansi-escapes@0.2.1
	strsim@0.11.1
	syn@2.0.119
	syn@3.0.6
	tempfile@3.27.0
	terminal_size@0.4.4
	termtree@0.5.1
	thiserror-impl@2.0.21
	thiserror@2.0.21
	toml_datetime@1.1.1+spec-1.1.0
	toml_edit@0.25.15+spec-1.1.0
	toml_parser@1.1.3+spec-1.1.0
	typenum@1.20.1
	unicode-ident@1.0.26
	unicode-width@0.1.14
	unicode-width@0.2.2
	uom@0.36.0
	utf8parse@0.2.2
	vte@0.14.1
	wait-timeout@0.2.1
	wasi@0.11.1+wasi-snapshot-preview1
	wasip2@1.0.4+wasi-0.2.12
	wasm-bindgen-macro-support@0.2.129
	wasm-bindgen-macro@0.2.129
	wasm-bindgen-shared@0.2.129
	wasm-bindgen@0.2.129
	windows-link@0.2.1
	windows-sys@0.59.0
	windows-sys@0.61.2
	windows-targets@0.52.6
	windows_aarch64_gnullvm@0.52.6
	windows_aarch64_msvc@0.52.6
	windows_i686_gnu@0.52.6
	windows_i686_gnullvm@0.52.6
	windows_i686_msvc@0.52.6
	windows_x86_64_gnu@0.52.6
	windows_x86_64_gnullvm@0.52.6
	windows_x86_64_msvc@0.52.6
	winnow@1.0.4
	wit-bindgen@0.57.1
	zerocopy-derive@0.8.58
	zerocopy@0.8.58
	zmij@1.0.23
"

CARGO_OPTIONAL=1
inherit shell-completion cargo toolchain-funcs

DESCRIPTION="A command-line benchmarking tool"
HOMEPAGE="https://github.com/sharkdp/hyperfine"
SRC_URI="
	https://github.com/sharkdp/${PN}/archive/v${PV}.tar.gz -> ${P}.tar.gz
	${CARGO_CRATE_URIS}
"

LICENSE="|| ( Apache-2.0 MIT )"
# Dependent crate licenses
LICENSE+=" Apache-2.0 BSD MIT MPL-2.0 Unicode-3.0"
SLOT="0"
KEYWORDS="~alpha"
BDEPEND=">=dev-lang/rust-alpha-bin-1.102.0_pre20261111:0"
RESTRICT="test"

QA_FLAGS_IGNORED="usr/bin/${PN}"

pkg_setup() {
 [[ ${MERGE_TYPE} == binary ]] && return 0
 case ${CHOST} in alpha*-linux-gnu*) ;; *) die "Native Alpha GNU/Linux only" ;; esac
 tc-is-cross-compiler && die "Native Alpha builds only"
}
src_unpack() {
 cargo_src_unpack
}

hyperfine_alpha_env() {
	export ALPHA_RUST_PREFIX="${BROOT}/opt/rust-alpha-experimental"
	export ALPHA_HYPERFINE_NIX="${WORKDIR}/alpha-nix"
	export CARGO="${T}/cargo-alpha-hyperfine"
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

src_prepare() {
 default
 cp -a "${BROOT}/opt/rust-alpha-experimental/share/rust-alpha-experimental/support/nix-0.31.3" "${WORKDIR}/alpha-nix" || die
 pushd "${WORKDIR}/alpha-nix" >/dev/null || die
 eapply "${FILESDIR}/nix-0.31.3-alpha-statfs.patch"
 popd >/dev/null || die
 cp "${FILESDIR}/cargo-alpha-hyperfine" "${T}/cargo-alpha-hyperfine" || die
 chmod +x "${T}/cargo-alpha-hyperfine" || die
 sed -i '/strip =/d' Cargo.toml || die
}
src_configure() {
 hyperfine_alpha_env
 "${CARGO}" generate-lockfile --offline || die
 "${CARGO}" metadata --locked --offline --format-version 1 > "${T}/alpha-cargo-metadata.json" || die
}
src_compile() {
 hyperfine_alpha_env
 "${CARGO}" build --release --locked --offline --bin hyperfine -j "$(makeopts_jobs)" || die
}
src_install() {
 dobin "${WORKDIR}/alpha-target/release/hyperfine"
 local build_dir="$(dirname $(find "${WORKDIR}/alpha-target" -name ${PN}.bash -print -quit))"
 newbashcomp "${build_dir}/${PN}.bash" "${PN}"
 dozshcomp "${build_dir}/_${PN}"
 dofishcomp "${build_dir}/${PN}.fish"
 doman doc/hyperfine.1
 einstalldocs
}
