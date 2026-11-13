Skim5.6.6-r1 candidate: CLI only; frizbee/image/listen disabled. Alpha uses the standard allocator. Native build and runtime validation pending. Source archive matches Gentoo hashes. Existing shared helper/patch bank from 20261113 maintenance required. No shell integration activation.

2026-11-13 10:48Z: Skim initial build failed exit1 in SDK Rustix1.1.5
signal.rs: nonexistent SIGSTKFLT. Preserved attempt1-failed.log locally.
New private rustix-1.1.5.patch adds Alpha to six STKFLT-exclusion/EMT-inclusion
guards (constants, raw conversion, Debug). Exact failing native rustc metadata
replay PASS (rustix-115-check.log); feature-scoped preflight rejection/acceptance
PASS. Guard only process/runtime/io_uring, not fs-only users. Recipe copies SDK
rustix-compiler privately and overrides plain rustix alias; installed SDK untouched.
Shared patch/checker source and publication mirror under 20261113-post-maintenance.
Launched build-skim-retry1.sh -j3 with unique skim-retry1-started/shared lock;
inspect skim-retry1-build.log,skim-retry1-driver.log,skim-retry1-exit-status.
Full Skim build/runtime pending; frozen maintenance/SDK archives unchanged.
This signal fix is a future SDK-support candidate after full application validation,
not a reason to silently replace installed or published support trees now.

Skim retry1 passed Rustix but failed nix0.28.0 fs accessors: Alpha fields are
signed c_long. Private nix0.28 patch preserves those types for blocks/free/available
rather than casting or panicking. Shared checker now requires private nix_028
for this version with fs enabled. Set patch.crates-io.nix_028.package="nix" before
alpha-rust_patch_crate nix 0.28.0 "${ECARGO_VENDOR}/nix-0.28.0" nix_028.
Failure evidence: work/skim-rust-es40/logs/retry1-failed.log.

2026-11-13 12:18Z: Skim5.6.6-r1 retry2 installed exit0, sk5.6.6. Native
runtime smoke PASS exact filtering, fuzzy ranking, Unicode, streamed records,
no-match status, PTY preview/selection and cancellation. Disposable fixture
skim-smoke-ndrkbx55 with terminal captures; no shell integration activated.
CLI only, frizbee/image/listen disabled, standard allocator. Both Rustix1.1.5
signals and nix0.28 signed-counter private fixes now passed full application
build and this smoke coverage (not full suites or exhaustive signal/ABI tests).
Local work/skim-rust-es40/logs and check-skim-runtime.py preserve evidence.
Frozen SDK/maintenance releases unchanged. New source remains post-maintenance.

Presenterm0.15.1-r1 candidate (Gentoo0.15.1 source hashes verified): terminal
Markdown presentations, native Oniguruma/image parsing paths. Uses upstream
empty default features (sixel/json-schema off), explicitly enables Crossterm
libc, existing private Rustix1.1.5 signal patch. Retains docs/themes/examples.
No server, shell integration or external code-execution configuration enabled.
Build/runtime pending. After install test disposable Markdown slides, Unicode,
syntax highlighting, next/previous navigation and quit; avoid executing slide
code and external renderers. Sources/logs work/presenterm-rust-es40.

Presenterm offline-resolution fix: Crossterm0.29/tempfile3.20 accept Rustix1.x,
but missing linux-raw-sys0.12 source metadata made Cargo skip SDK Rustix1.1.5
and choose vendored1.0.7 (whose raw dependency remained active). Add
linux-raw-sys@0.12.1 to CRATES with registry-verified archive hashes. Cargo may
need archives for inactive target dependencies to resolve a patched crate.
This does NOT enable direct syscalls: Alpha-filtered normal/build graph verified
private Rustix1.1.5 and no linux-raw-sys; shared preflight PASS. Test the graph,
not merely a warning that a patch is unused. Installed SDK unchanged.
Evidence: work/presenterm-rust-es40/reference/presenterm-resolver-fixed.log,
failed-metadata.json and presenterm-probe-metadata.json; initial failure log in
logs/attempt1-failed.log. Full application build/runtime validation pending.

2026-11-13 14:18Z: Presenterm0.15.1-r1 retry1 installed exit0, version0.15.1.
Native PTY smoke PASS Markdown/Unicode rendering, next slide with colored Rust
syntax snippet, previous slide and clean quit. Fixture presenterm-smoke-hxakb71a,
terminal.ansi preserved. No snippet execution, external renderers, images/sixel
or full-suite validation. Runtime script work/presenterm-rust-es40/check-presenterm-runtime.py;
logs copied locally. Offline resolver fix now validated through full native
application build/runtime smoke. Installed SDK and frozen archives unchanged.
No next build launched. Next heartbeat can evaluate another useful scoped app.

git-absorb0.9.0-r1 candidate: native libgit2-backed index/diff/commit operations.
Gentoo source hashes verified; preserves system libgit2 1.9 requirement, docs
and generated shell completions. No integration activated. Build/runtime not
yet validated. Smoke MUST use a freshly created disposable Git repository only:
check dry-run leaves HEAD/index unchanged, fixup commit matches staged edit,
working tree/index state and no-change behavior. Never run against user repos.
Refs/logs work/git-absorb-rust-es40. No network Git operations needed.

git-absorb initial preflight found clap_builder4.5.2 -> terminal_size0.3.0 ->
Rustix0.38.32 -> linux-raw-sys0.4.13. Tempfile3.10.1 also appears in metadata
but is a development-only root dependency. Private terminal_size0.3 patch uses
libc isatty/ioctl, preserving positive-size checks and None on failures; removes
its Rustix edge. Same approach as earlier terminal_size0.2.6/0.4.1, exact-version
adaptation. Shared checker now requires the private0.3 copy. Runtime/native
probe and application preflight evidence recorded under work/git-absorb-rust-es40.
Application installation/runtime validation still pending.
