# Private experimental Alpha build, based on the upstream Gentoo package.
EAPI=8
ALPHA_CARGO_BIN=fd
ALPHA_SOURCE_ARCHIVE="fd-10.5.0-alpha-experimental-20261111.tar.xz"
inherit alpha-cargo
DESCRIPTION="File search, private Alpha GCC-backend build"
HOMEPAGE="https://github.com/sharkdp/fd"
SRC_URI="${ALPHA_SOURCE_ARCHIVE}"
S="${WORKDIR}/${PN}-${PV}"
LICENSE="|| ( Apache-2.0 MIT ) MIT Unicode-3.0 ZLIB"
SLOT="0"
KEYWORDS="~alpha"
# Uses upstream's default completions feature and the system allocator.
RDEPEND=">=sys-libs/glibc-2.39"
QA_FLAGS_IGNORED="usr/bin/fd"

src_install() {
	alpha-cargo_src_install
	doman doc/*.1
	local binary=$(alpha-cargo_binary)
	"${binary}" --gen-completions bash > "${T}/fd.bash" || die
	insinto /usr/share/bash-completion/completions
	newins "${T}/fd.bash" fd
	"${binary}" --gen-completions fish > "${T}/fd.fish" || die
	insinto /usr/share/fish/vendor_completions.d
	doins "${T}/fd.fish"
	insinto /usr/share/zsh/site-functions
	doins contrib/completion/_fd
}
