# Private experimental Alpha build, based on the upstream Gentoo package.
EAPI=8
ALPHA_CARGO_BIN=zoxide
ALPHA_SOURCE_ARCHIVE="zoxide-0.10.0-alpha-experimental-20261111.tar.xz"
inherit alpha-cargo
DESCRIPTION="Directory navigation helper, private Alpha GCC-backend build"
HOMEPAGE="https://github.com/ajeetdsouza/zoxide"
SRC_URI="${ALPHA_SOURCE_ARCHIVE}"
S="${WORKDIR}/${PN}-${PV}"
LICENSE="MIT MPL-2.0 Unicode-3.0 ZLIB || ( Apache-2.0 CC0-1.0 MIT-0 )"
SLOT="0"
KEYWORDS="~alpha"
RDEPEND=">=sys-libs/glibc-2.39"
QA_FLAGS_IGNORED="usr/bin/zoxide"

src_install() {
	alpha-cargo_src_install
	doman man/man1/*
	dodoc CHANGELOG.md
	insinto /usr/share/bash-completion/completions
	newins contrib/completions/zoxide.bash zoxide
	insinto /usr/share/zsh/site-functions
	doins contrib/completions/_zoxide
	insinto /usr/share/fish/vendor_completions.d
	doins contrib/completions/zoxide.fish
	insinto /usr/share/zoxide
	doins init.fish zoxide.plugin.zsh
}
