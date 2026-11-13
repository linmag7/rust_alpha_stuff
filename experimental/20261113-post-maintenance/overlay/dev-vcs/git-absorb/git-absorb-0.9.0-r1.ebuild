# Copyright 2025-2026 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=8

CRATES=""
CARGO_OPTIONAL=1
inherit cargo alpha-rust shell-completion

DESCRIPTION="Automatically absorb staged changes into git current branch"
HOMEPAGE="https://github.com/tummychow/git-absorb"
SRC_URI="https://github.com/tummychow/${PN}/archive/refs/tags/${PV}.tar.gz -> ${P}.tar.gz"
SRC_URI+=" https://github.com/gentoo-crate-dist/${PN}/releases/download/${PV}/${P}-crates.tar.xz"

LICENSE="BSD"
# Dependent crate licenses
LICENSE+=" Apache-2.0 MIT Unicode-DFS-2016"
SLOT="0"
KEYWORDS="~alpha"
RESTRICT="test"
QA_PRESTRIPPED="usr/bin/git-absorb"

RDEPEND="
	>=dev-libs/libgit2-1.9:=
	<dev-libs/libgit2-1.10
"
DEPEND="${RDEPEND}"
BDEPEND="app-text/asciidoc >=dev-lang/rust-alpha-bin-1.102.0_pre20261111-r2:0"

DOCS=( README.md )

QA_FLAGS_IGNORED="usr/bin/${PN}"

pkg_setup() { alpha-rust_pkg_setup; }
src_unpack() { cargo_src_unpack; }
src_prepare() {
 default
 alpha-rust_prepare
 alpha-rust_patch_crate terminal_size 0.3.0 "${ECARGO_VENDOR}/terminal_size-0.3.0"
 alpha-rust_finish_prepare
}
src_configure() { alpha-rust_configure; }
src_compile() {
	export LIBGIT2_NO_VENDOR=1

	alpha-rust_build --bin git-absorb

	GIT_ABSORB_BIN="${WORKDIR}/alpha-target/release/${PN}"

	emake -C Documentation

	# Prepare shell completion generation
	mkdir completions || die
	local shell
	for shell in bash fish zsh; do
		"${GIT_ABSORB_BIN}" --gen-completions \
					 ${shell} \
					 > completions/${PN}.${shell} \
			|| die
	done
}

src_install() {
	dobin "${WORKDIR}/alpha-target/release/git-absorb"
	doman Documentation/${PN}.1

	newbashcomp "completions/${PN}.bash" "${PN}"
	dofishcomp "completions/${PN}.fish"
	dozshcomp "completions/${PN}.zsh"

	default
}
