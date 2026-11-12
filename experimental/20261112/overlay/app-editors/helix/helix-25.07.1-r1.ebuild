# Copyright 2024-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

CARGO_OPTIONAL=1
CRATES="linux-raw-sys@0.12.0"

RUST_MIN_VER="1.82"

inherit cargo desktop shell-completion xdg

DESCRIPTION="A post-modern text editor"
HOMEPAGE="
	https://helix-editor.com/
	https://github.com/helix-editor/helix
"
SRC_URI="
	https://github.com/helix-editor/helix/releases/download/${PV}/${P}-source.tar.xz -> ${P}.tar.xz
	https://github.com/gentoo-crate-dist/helix/releases/download/${PV}/${P}-crates.tar.xz
	${CARGO_CRATE_URIS}
"
S="${WORKDIR}"

LICENSE="MPL-2.0"
# Dependent crate licenses
LICENSE+="
	Apache-2.0 BSD Boost-1.0 ISC MIT MPL-2.0 MPL-2.0 Unicode-DFS-2016
	ZLIB
"
SLOT="0"
KEYWORDS="~alpha"
IUSE="+grammar"

BDEPEND=">=dev-lang/rust-alpha-bin-1.102.0_pre20261111:0"
RDEPEND="dev-vcs/git"
RESTRICT="test"

pkg_setup() {
	QA_FLAGS_IGNORED="
		usr/bin/hx
		usr/$(get_libdir)/${PN}/.*\.so
	"
	export HELIX_DEFAULT_RUNTIME="${EPREFIX}/usr/share/${PN}/runtime"
	use grammar || export HELIX_DISABLE_AUTO_GRAMMAR_BUILD=1
	[[ ${MERGE_TYPE} == binary ]] && return 0
	case ${CHOST} in alpha*-linux-gnu*) ;; *) die "Native Alpha GNU/Linux only" ;; esac
	tc-is-cross-compiler && die "Native Alpha builds only"
	[[ -x ${BROOT}/opt/rust-alpha-experimental/bin/cargo-alpha ]] || die "Install rust-alpha-bin first"
}

helix_alpha_env() {
	export ALPHA_RUST_PREFIX="${BROOT}/opt/rust-alpha-experimental"
	export CARGO="${ALPHA_RUST_PREFIX}/bin/cargo-alpha"
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
	export RUSTFLAGS="-Cdebuginfo=0 -Clink-arg=-B${T}/alpha-crt/"
}

src_unpack() {
	cargo_src_unpack
}

src_prepare() {
	default
	# Use Crossterm's existing libc backend instead of its old rustix backend.
	cp -a "${ECARGO_VENDOR}/crossterm-0.28.1" "${WORKDIR}/alpha-crossterm" || die
	pushd "${WORKDIR}/alpha-crossterm" >/dev/null || die
	eapply "${FILESDIR}/crossterm-0.28.1-libc-backend.patch"
	popd >/dev/null || die
	cat >> "${ECARGO_HOME}/config.toml" <<-EOF || die
	[patch.crates-io]
	crossterm = { path = "${WORKDIR}/alpha-crossterm" }
	EOF
}

src_configure() {
	helix_alpha_env
	"${CARGO}" update --offline -p libc --precise 0.2.191 || die
	"${CARGO}" update --offline -p rustix@1.0.7 || die
	"${CARGO}" metadata --locked --offline --features crossterm/libc --format-version 1 > "${T}/alpha-cargo-metadata.json" || die
}

src_compile() {
	helix_alpha_env
	"${CARGO}" build --release --locked --offline -p helix-term --features crossterm/libc --bin hx -j "$(makeopts_jobs)" || die
}

src_install() {
	dobin "${WORKDIR}/alpha-target/release/hx"

	insinto "/usr/$(get_libdir)/${PN}"
	use grammar && doins runtime/grammars/*.so
	rm -r runtime/grammars || die
	use grammar && dosym -r "/usr/$(get_libdir)/${PN}" "/usr/share/${PN}/runtime/grammars"

	insinto /usr/share/helix
	doins -r runtime

	doicon -s 256x256 contrib/${PN}.png
	domenu contrib/Helix.desktop

	insinto /usr/share/metainfo
	doins contrib/Helix.appdata.xml

	newbashcomp contrib/completion/hx.bash hx
	newzshcomp contrib/completion/hx.zsh _hx
	dofishcomp contrib/completion/hx.fish

	DOCS=(
		README.md
		CHANGELOG.md
		docs/
	)
	HTML_DOCS=(
		book/
	)
	einstalldocs
}

pkg_postinst() {
	if ! use grammar ; then
		einfo "Grammars are not installed yet. To fetch them, run:"
		einfo ""
		einfo "  hx --grammar fetch && hx --grammar build"
	fi

	xdg_desktop_database_update
	xdg_icon_cache_update
}

pkg_postrm() {
	xdg_desktop_database_update
	xdg_icon_cache_update
}
