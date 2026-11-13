# Copyright 2017-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

CRATES=""
CARGO_OPTIONAL=1

inherit cargo alpha-rust optfeature shell-completion

DESCRIPTION="Command-line fuzzy finder"
HOMEPAGE="https://github.com/skim-rs/skim"
SRC_URI="
	https://github.com/skim-rs/skim/archive/v${PV}.tar.gz -> ${P}.gh.tar.gz
	https://github.com/gentoo-crate-dist/${PN}/releases/download/v${PV}/${P}-crates.tar.xz
	${CARGO_CRATE_URIS}
"

LICENSE="MIT"
# Dependent crate licenses
LICENSE+="
	0BSD Apache-2.0 LGPL-3 MIT Unicode-3.0 Unicode-DFS-2016 WTFPL-2 ZLIB
"
SLOT="0"
KEYWORDS="~alpha"
IUSE=""

RESTRICT="test"

BDEPEND=" >=dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2:0"

QA_FLAGS_IGNORED="usr/bin/sk"

DOCS=(
	ARCHITECTURE.md
	CHANGELOG.md
	README.md
)

PATCHES=( "${FILESDIR}/skim-5.6.6-alpha-system-allocator.patch" )
QA_PRESTRIPPED="usr/bin/sk"
pkg_setup() { alpha-rust_pkg_setup; }
src_unpack() { cargo_src_unpack; }
src_prepare() {
 default
 alpha-rust_prepare
 alpha-rust_patch_crate nix 0.31.3 "${BROOT}/opt/rust-alpha-experimental/share/rust-alpha-experimental/support/nix-0.31.3" nix_031
 alpha-rust_patch_crate rustix 1.1.5 "${BROOT}/opt/rust-alpha-experimental/share/rust-alpha-experimental/support/rustix-compiler"
 printf 'args+=( --config %q )\n' 'patch.crates-io.nix_028.package="nix"' >> "${T}/alpha-rust-cargo" || die
 alpha-rust_patch_crate nix 0.28.0 "${ECARGO_VENDOR}/nix-0.28.0" nix_028
 alpha-rust_finish_prepare
}
src_configure() { alpha-rust_configure --no-default-features --features cli; }
src_compile() { alpha-rust_build --bin sk --no-default-features --features cli; }

src_install() {
	# prevent cargo_src_install() blowing up on man installation
	mv man manpages || die

	dobin "${WORKDIR}/alpha-target/release/sk"

	einstalldocs
	doman manpages/man1/*

	dobin bin/sk-tmux
	insinto /usr/share/vim/vimfiles/plugin
	doins plugin/skim.vim

	# install shell keybindings
	insinto "/usr/share/${PN}"
	doins shell/key-bindings.*

	newbashcomp shell/completion.bash sk
	newfishcomp shell/completion.fish sk.fish
	newzshcomp shell/completion.zsh _sk
}

pkg_postinst() {
	optfeature "sk-tmux integration" app-misc/tmux
	optfeature "vim plugin integration" app-editors/vim app-editors/gvim
}
