Configuration for the experimental Git 2.56.1-r1 overlay ebuild.

Install package.use as /etc/portage/package.use/rust-alpha-git and
package.accept_keywords as /etc/portage/package.accept_keywords/rust-alpha-git.
Install package.use.mask under the local profile at
/etc/portage/profile/package.use.mask/rust-alpha-git to remove the Rust USE
mask only for this exact overlay package version. Merge carefully if the
corresponding config path is a file rather than a directory.

The current requested configuration enables rust and keyring, disables test,
and uses the native /opt/rust-alpha-experimental toolchain. The full Git test
suite was explicitly waived by the user for this build.
