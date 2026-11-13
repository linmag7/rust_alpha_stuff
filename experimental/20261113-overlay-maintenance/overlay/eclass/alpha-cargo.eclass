# Copyright 2026 Gentoo Authors and private Alpha experiment contributors
# Distributed under the terms of the GNU General Public License v2
# Private experimental native Alpha builds; not a replacement for cargo.eclass.

case ${EAPI} in 8) ;; *) die "alpha-cargo requires EAPI 8" ;; esac

BDEPEND="${BDEPEND} >=dev-lang/rust-alpha-bin-1.102.0_pre20261111:0"
RESTRICT="${RESTRICT} fetch mirror"

EXPORT_FUNCTIONS pkg_setup pkg_nofetch src_configure src_compile src_test src_install

alpha-cargo_pkg_nofetch() {
	einfo "Private experiment: copy ${ALPHA_SOURCE_ARCHIVE} from the prepared"
	einfo "bundle into your Portage DISTDIR, then retry emerge."
}

alpha-cargo_pkg_setup() {
	[[ ${MERGE_TYPE} == binary ]] && return 0
	case ${CHOST} in alpha*-linux-gnu*) ;; *) die "This private eclass supports Alpha GNU/Linux only" ;; esac
	: "${ALPHA_RUST_PREFIX:=${BROOT}/opt/rust-alpha-experimental}"
	[[ -x ${ALPHA_RUST_PREFIX}/bin/cargo-alpha ]] || die "Install dev-lang/rust-alpha-bin first"
	export CARGO="${ALPHA_RUST_PREFIX}/bin/cargo-alpha"
	export RUSTC="${ALPHA_RUST_PREFIX}/bin/rustc"
}

alpha-cargo_env() {
	# The packaged launcher selects libgccjit, native GCC, panic=abort and patched crates.
	# Use a fixed tested configuration until general Gentoo RUSTFLAGS are validated.
	unset CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS CARGO_BUILD_TARGET
	unset RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
	export RUSTFLAGS="-Cdebuginfo=0"
	export CARGO_HOME="${T}/cargo-home" CARGO_TARGET_DIR="${WORKDIR}/target"
	export CARGO_INCREMENTAL=0 CARGO_NET_OFFLINE=true
	export CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_STRIP=false
	export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
	export CARGO_PROFILE_RELEASE_PANIC=abort
	export TMPDIR="${T}"
	mkdir -p "${CARGO_HOME}" || die
}

alpha-cargo_src_configure() {
	alpha-cargo_env
	[[ -f .cargo/config.toml && -d vendor ]] || die "Prepared offline source bundle is required"
	"${CARGO}" metadata --locked --offline --format-version 1 > "${T}/cargo-metadata.json" || die
}

alpha-cargo_src_compile() {
	alpha-cargo_env
	"${CARGO}" build --release --locked --offline \
		--target alphaev6-unknown-linux-gnu --bin "${ALPHA_CARGO_BIN}" \
		-j "${ALPHA_RUST_JOBS:-2}" || die
}

alpha-cargo_binary() {
	echo "${WORKDIR}/target/alphaev6-unknown-linux-gnu/release/${ALPHA_CARGO_BIN}"
}

alpha-cargo_src_test() {
	alpha-cargo_env
	bash "${FILESDIR}/smoke.sh" "$(alpha-cargo_binary)" "${T}/smoke" || die "Application smoke test failed"
}

alpha-cargo_src_install() {
	dobin "$(alpha-cargo_binary)"
	dodoc README.md
}
