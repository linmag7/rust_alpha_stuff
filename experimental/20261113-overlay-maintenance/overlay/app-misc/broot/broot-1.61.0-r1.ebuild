# Copyright 2021-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

RUST_MIN_VER="1.90.0"
CARGO_OPTIONAL=1
inherit cargo shell-completion alpha-rust
BDEPEND+=" >=dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2:0"

DESCRIPTION="A new way to see and navigate directory trees"
HOMEPAGE="https://dystroy.org/broot/ https://github.com/Canop/broot"
SRC_URI="https://github.com/Canop/broot/archive/v${PV}.tar.gz -> ${P}.tar.gz"
SRC_URI+=" https://github.com/gentoo-crate-dist/broot/releases/download/v${PV}/${P}-crates.tar.xz"

LICENSE="Apache-2.0"
# Dependent crate licenses
LICENSE+="
	Apache-2.0 Apache-2.0-with-LLVM-exceptions BSD-2 BSD Boost-1.0
	LGPL-3+ MIT MPL-2.0 UoI-NCSA Unicode-3.0 ZLIB
"
SLOT="0"
KEYWORDS="~alpha"
RESTRICT="test"
IUSE="X"

DEPEND="
	dev-db/sqlite:3
	virtual/zlib:=
	X? ( x11-libs/libxcb:= )
"
RDEPEND="${DEPEND}"

QA_FLAGS_IGNORED="usr/bin/${PN}"

pkg_setup() { alpha-rust_pkg_setup; }
src_unpack() { cargo_src_unpack; }

src_prepare() {
	default
	alpha-rust_prepare
	alpha-rust_patch_crate which 4.4.2 "${ECARGO_VENDOR}/which-4.4.2"
	alpha-rust_patch_crate lfs-core 0.20.0 "${ECARGO_VENDOR}/lfs-core-0.20.0"
	alpha-rust_finish_prepare

	local mandate=$(date -r man/page +'%Y/%m/%d' || die)
	sed -e "s|#version|${PV}|" \
		-e "s|#date|${mandate}|" \
		man/page > "${T}"/${PN}.1 || die
}

src_configure() {
 export LIBSQLITE3_SYS_USE_PKG_CONFIG=1
 local features=sixel
 use X && features+=,clipboard
 alpha-rust_configure --no-default-features --features "${features}"
}
src_compile() {
 export LIBSQLITE3_SYS_USE_PKG_CONFIG=1
 local features=sixel
 use X && features+=,clipboard
 alpha-rust_build --bin broot --no-default-features --features "${features}"
}

src_install() {
	dobin "${WORKDIR}/alpha-target/release/broot"

	doman "${T}"/${PN}.1

	local build_dir=( "${WORKDIR}"/alpha-target/release/build/${PN}/*/out "${WORKDIR}"/alpha-target/release/build/${PN}-*/out )
	local completions_dir=
	for candidate in "${build_dir[@]}"; do
		[[ -f ${candidate}/${PN}.bash ]] && completions_dir=${candidate}
	done
	[[ -n ${completions_dir} ]] || die "Missing generated completions"
	cd "${completions_dir}" || die

	newbashcomp ${PN}.bash ${PN}
	newbashcomp br.bash br

	dozshcomp _${PN} _br
	dofishcomp ${PN}.fish br.fish
}
