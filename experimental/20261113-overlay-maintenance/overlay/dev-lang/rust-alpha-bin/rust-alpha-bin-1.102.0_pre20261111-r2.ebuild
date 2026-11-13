# Private experimental binary toolchain. Not official Gentoo Rust support.
EAPI=8
DESCRIPTION="Private experimental Alpha-native Rust/GCC/binutils toolchain"
HOMEPAGE="https://github.com/rust-lang/rustc_codegen_gcc"
TOOLCHAIN_ARCHIVE="alpha-native-toolchain-20261112-r2-experimental.tar.xz"
SRC_URI="${TOOLCHAIN_ARCHIVE}"
S="${WORKDIR}"
LICENSE="GPL-3+ LGPL-2.1+ MIT Apache-2.0 BSD || ( GPL-3 LGPL-3 )"
SLOT="0"
KEYWORDS="~alpha"
RESTRICT="fetch mirror strip"
RDEPEND=">=sys-libs/glibc-2.39"
QA_PREBUILT="opt/rust-alpha-experimental/*"

pkg_nofetch() {
	einfo "Copy ${TOOLCHAIN_ARCHIVE} into your Portage DISTDIR."
	einfo "This is a local, private proof of concept; no public download is provided."
}
src_compile() { :; }
src_install() {
	docompress -x /opt/rust-alpha-experimental
	dodir /opt
	cp -a "${S}/opt/rust-alpha-experimental" "${ED}/opt/" || die
	# Compiler users link .so/.a files directly; omit unused libtool metadata.
	rm -f "${ED}"/opt/rust-alpha-experimental/lib/*.la || die
	(
		cd "${ED}/opt/rust-alpha-experimental" || exit 1
		find . -type f ! -name MANIFEST.sha256 -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > MANIFEST.sha256
	) || die "Cannot generate installed file checksums"
}
pkg_postinst() {
	elog "Installed experimental Alpha toolchain under /opt/rust-alpha-experimental."
	elog "The rust-alpha overlay selects it for its application ebuilds."
	elog "Interactive use: /opt/rust-alpha-experimental/bin/cargo-alpha"
	elog "Other Gentoo Rust ebuilds still need individual integration and validation."
}
