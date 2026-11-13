# Copyright 2021-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2
EAPI=8
RUST_MIN_VER=1.87.0
CARGO_OPTIONAL=1
inherit cargo alpha-rust
BDEPEND+=" >=dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2:0"
DESCRIPTION="Create a book from markdown files"
HOMEPAGE="https://rust-lang.github.io/mdBook/"
SRC_URI="https://github.com/rust-lang/mdBook/archive/refs/tags/v${PV}.tar.gz -> ${P}.tar.gz
 https://distfiles.gentoo.org/pub/dev/ionen@gentoo.org/${P}-vendor.tar.xz"
S=${WORKDIR}/${P/b/B}
LICENSE="MPL-2.0 OFL-1.1 Apache-2.0 BSD CC-BY-4.0 CC0-1.0 ISC MIT Unicode-3.0"
SLOT="0"
KEYWORDS="~alpha"
RESTRICT="test"
QA_FLAGS_IGNORED="usr/bin/${PN}"
pkg_setup() { alpha-rust_pkg_setup; }
src_unpack() { cargo_src_unpack; }
src_prepare() {
 default
 alpha-rust_prepare
 alpha-rust_finish_prepare
}
src_configure() { alpha-rust_configure --no-default-features --features watch,search; }
src_compile() { alpha-rust_build --bin mdbook --no-default-features --features watch,search; }
src_install() {
 dobin "${WORKDIR}/alpha-target/release/mdbook"
 dodoc CHANGELOG.md README.md
}
