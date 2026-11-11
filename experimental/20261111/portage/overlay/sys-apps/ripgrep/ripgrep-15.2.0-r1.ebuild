# Private experimental Alpha build, based on the upstream Gentoo package.
EAPI=8
ALPHA_CARGO_BIN=rg
ALPHA_SOURCE_ARCHIVE="ripgrep-15.2.0-alpha-experimental-20261111.tar.xz"
inherit alpha-cargo
DESCRIPTION="Recursive text search, private Alpha GCC-backend build"
HOMEPAGE="https://github.com/BurntSushi/ripgrep"
SRC_URI="${ALPHA_SOURCE_ARCHIVE}"
S="${WORKDIR}/${PN}-${PV}"
LICENSE="Apache-2.0 BSD MIT Unicode-3.0"
SLOT="0"
KEYWORDS="~alpha"
# PCRE2 is deliberately not enabled in this first validated configuration.
RDEPEND=">=sys-libs/glibc-2.39"
QA_FLAGS_IGNORED="usr/bin/rg"

src_install() {
	alpha-cargo_src_install
	local binary=$(alpha-cargo_binary)
	"${binary}" --generate man > "${T}/rg.1" || die
	doman "${T}/rg.1"
	"${binary}" --generate complete-bash > "${T}/rg.bash" || die
	insinto /usr/share/bash-completion/completions
	newins "${T}/rg.bash" rg
	"${binary}" --generate complete-zsh > "${T}/_rg" || die
	insinto /usr/share/zsh/site-functions
	doins "${T}/_rg"
	"${binary}" --generate complete-fish > "${T}/rg.fish" || die
	insinto /usr/share/fish/vendor_completions.d
	doins "${T}/rg.fish"
	dodoc CHANGELOG.md FAQ.md GUIDE.md
}
