Alpha Rust overlay maintenance — 2026-11-13
==========================================
Validated on the ES40 with SDK r2: all four new applications installed and
passed the runtime smoke checks described below. No full test-suite claim.

Requires the existing experimental SDK revision 2 and its configured rust-alpha
Portage repository. This is an overlay/configuration update, not an SDK binary
upgrade. The complete overlay combines the frozen r2 recipes with validated
post-r2 adaptations. No compiler, libc support tree or release asset is replaced.

New package recipes:
  dev-vcs/gitui 0.28.1-r1: terminal Git status/history navigation
  app-misc/broot 1.61.0-r1: tree navigation/search/text preview; -X, sixel
  app-text/mdbook 0.5.4-r2: HTML/search/watch; serve disabled
  app-misc/jaq 3.1.2-r1: JSON query interpreter; mimalloc disabled

Apply on the Alpha machine as root from this directory after validation:
  sha256sum -c SHA256SUMS
  test "$(uname -m)" = alpha || exit 1
  test -d /var/db/repos/rust-alpha || exit 1
  test -x /opt/rust-alpha-experimental/bin/rustc || exit 1
  backup=/root/rust-alpha-overlay-backup-$(date +%Y%m%d-%H%M%S)
  mkdir "$backup"
  cp -a /var/db/repos/rust-alpha "$backup/overlay"
  cp -a /etc/portage/package.accept_keywords /etc/portage/package.unmask /etc/portage/package.use "$backup/"
  cp -a overlay/. /var/db/repos/rust-alpha/

Review the exact entries under config/ and copy only packages you intend to
build into the corresponding /etc/portage/package.accept_keywords/ and
/etc/portage/package.unmask/ directories. Also copy Broot's -X entry into
/etc/portage/package.use/ if selecting Broot. Do not replace the whole directories.
For example, with the jaq entries installed:
  emerge --pretend --verbose '=app-misc/jaq-3.1.2-r1::rust-alpha'
  MAKEOPTS=-j3 emerge --ask --verbose '=app-misc/jaq-3.1.2-r1::rust-alpha'

Inspect the plan: do not accept unrelated upgrades or replacement compilers.
Existing packages need not be rebuilt merely to install these overlay files.
Manifests follow thin-manifests=true: upstream DIST hashes are preserved; stale
EBUILD/MISC entries from the source repository are omitted.
Keep the backup for rollback and do not apply this during an active build.

Read RUST-BUILD-RECIPE.txt for reusable dependency fixes and limitations.
Runtime smoke checks are narrower than package test suites; these remain
experimental personal-fork recipes with tests restricted. Broot image/sixel
and clipboard paths are untested; no mdBook server was started. Floating-point,
unwinding and arbitrary ABI combinations have not been exhaustively validated.
