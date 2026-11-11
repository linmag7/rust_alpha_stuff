#!/usr/bin/env python3
"""Cargo integration and runners for the relocatable Alpha SDK."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'alphaev6-unknown-linux-gnu'
HOST = 'x86_64-unknown-linux-gnu'


def fail(message):
    print('alpha: ' + message, file=sys.stderr)
    raise SystemExit(2)


def config():
    path = ROOT / 'config.json'
    return json.loads(path.read_text()) if path.exists() else {}


def environment():
    env = os.environ.copy()
    cache = Path(env.get('ALPHA_CACHE_DIR', ROOT / 'var')).resolve()
    build_id = (ROOT / 'build-id').read_text().strip() if (ROOT / 'build-id').exists() else 'legacy'
    for name in ('tmp', 'cargo-home', 'target'):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env.update(TMPDIR=str(cache / 'tmp'), CARGO_HOME=str(cache / 'cargo-home'),
               RUSTC=str(ROOT / 'bin/alpha-rustc'),
               RUSTDOC=str(ROOT / 'bin/alpha-rustdoc'),
               CARGO_TARGET_DIR=env.get('CARGO_TARGET_DIR', str(cache / 'target' / build_id)),
               CG_GCC_LTO1=str(ROOT / 'gcc/libexec/gcc' / TARGET / '16.0.0/lto1'),
               ALPHA_SYSROOT=str(ROOT / 'sysroot'),
               ALPHA_CC=str(ROOT / 'bin/alpha-cc'), ALPHA_AR=str(ROOT / 'bin/alpha-ar'))
    env['PATH'] = os.pathsep.join([str(ROOT / 'bin'), str(ROOT / 'gcc/bin'), env.get('PATH', '')])
    env['LD_LIBRARY_PATH'] = str(ROOT / 'toolchain/lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    for suffix in (TARGET, TARGET.replace('-', '_')):
        env['CC_' + suffix] = str(ROOT / 'bin/alpha-cc')
        env['AR_' + suffix] = str(ROOT / 'bin/alpha-ar')
    env.pop('CARGO_TARGET_ALPHAEV6_UNKNOWN_LINUX_GNU_RUNNER', None)
    env.setdefault('QEMU_CPU', 'ev6')
    return env


def execute(command, env=None):
    os.execvpe(str(command[0]), list(map(str, command)), env or os.environ)


def tool(name, args):
    env = environment()
    target_compile = name == 'alpha-cc' or any(TARGET in arg for arg in args)
    if target_compile:
        env['GCC_EXEC_PREFIX'] = str(ROOT / 'gcc/lib/gcc') + '/'
        env['COMPILER_PATH'] = os.pathsep.join(map(str, [ROOT / 'gcc/bin', ROOT / 'gcc/libexec/gcc' / TARGET / '16.0.0']))
    else:
        env.pop('GCC_EXEC_PREFIX', None)
        env.pop('COMPILER_PATH', None)
    if name in ('alpha-rustc', 'alpha-rustdoc'):
        binary = ROOT / 'toolchain/bin' / name.removeprefix('alpha-')
        if not any(a == '--sysroot' or a.startswith('--sysroot=') for a in args):
            args = ['--sysroot', str(ROOT / 'toolchain'), *args]
        execute([binary, *args], env)
    binary = 'ar' if name == 'alpha-ar' else 'gcc'
    flags = [] if binary == 'ar' else ['--sysroot=' + str(ROOT / 'sysroot')]
    execute([ROOT / 'gcc/bin' / (TARGET + '-' + binary), *flags, *args], env)


def doctor(verify=False):
    errors = []
    metadata = json.loads((ROOT / 'manifest.json').read_text())
    print('Alpha SDK ' + metadata['version'])
    print('SDK: ' + str(ROOT))
    print('Target: EV6+, Alpha Linux, glibc 2.41, panic=abort')
    if platform.machine() != 'x86_64' or sys.platform != 'linux':
        errors.append('This SDK requires an x86_64 Linux host')
    for command in ('python3', 'cc', 'ssh', 'tar'):
        found = shutil.which(command)
        print(command + ': ' + (found or 'MISSING'))
        if not found:
            errors.append('Install ' + command)
    env = environment()
    for label, command in (
        ('rustc', [ROOT / 'bin/alpha-rustc', '-V']),
        ('cargo', [ROOT / 'toolchain/bin/cargo', '-V']),
        ('gcc', [ROOT / 'bin/alpha-cc', '-dumpfullversion']),
        ('qemu', [ROOT / 'bin/qemu-alpha-static', '--version']),
        ('sysroot', [ROOT / 'bin/qemu-alpha-static', '-L', ROOT / 'sysroot', ROOT / 'sysroot/usr/lib/libc.so.6.1']),
    ):
        result = subprocess.run(list(map(str, command)), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(label + ': ' + (result.stdout.splitlines()[0] if result.stdout else '(no output)'))
        if result.returncode:
            errors.append(label + ' failed')
    for path in ROOT.rglob('*'):
        if path.is_symlink() and 'var' not in path.relative_to(ROOT).parts:
            if not path.exists() or not path.resolve().is_relative_to(ROOT):
                errors.append('External or broken symlink: ' + str(path.relative_to(ROOT)))
    if verify:
        for relative, expected in metadata['sha256'].items():
            path = ROOT / relative
            digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
            if digest != expected:
                errors.append('Checksum mismatch: ' + relative)
        print('Verified ' + str(len(metadata['sha256'])) + ' packaged files')
    if errors:
        fail('\n'.join(errors))
    print('SDK checks: PASS')


def runner(args):
    if not args:
        fail('runner requires an executable')
    env = environment()
    binary = Path(args[0]).resolve()
    if not binary.is_file():
        fail('Executable not found: ' + str(binary))
    kind = env.get('ALPHA_RUNNER', 'qemu')
    if kind == 'qemu':
        env['QEMU_LD_PREFIX'] = str(ROOT / 'sysroot')
        options = []
        if env.get('ALPHA_TEST') == '1':
            # libtest restarts argv[0] for aborting tests. A host launcher avoids
            # requiring a global binfmt registration for those subprocesses.
            identity = hashlib.sha256(str(binary).encode()).hexdigest()[:24]
            launcher = Path(env['TMPDIR']) / ('alpha-test-' + identity)
            command = shlex.join([str(ROOT / 'bin/qemu-alpha-static'), '-0', str(launcher),
                                  '-L', str(ROOT / 'sysroot'), str(binary)])
            with tempfile.NamedTemporaryFile(mode='w', dir=env['TMPDIR'], delete=False) as stream:
                stream.write('#!/bin/sh\nexec ' + command + ' "$@"\n')
                staging = Path(stream.name)
            staging.chmod(0o700)
            staging.replace(launcher)
            options = ['-0', launcher]
        execute([ROOT / 'bin/qemu-alpha-static', *options, '-L', ROOT / 'sysroot', binary, *args[1:]], env)
    if kind != 'es40':
        fail('Unknown runner: ' + kind)
    settings = config().get('es40', {})
    target = env.get('ALPHA_SSH_TARGET', settings.get('target', ''))
    local_user = env.get('ALPHA_SSH_LOCAL_USER', settings.get('local_user', ''))
    if local_user.startswith('-'):
        fail('Invalid local SSH user')
    if not target or target.startswith('-'):
        fail('Set ALPHA_SSH_TARGET or configure es40.target in SDK config.json')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', target]
    remote = '/tmp/alpha-sdk-' + uuid.uuid4().hex
    # Transfer the executable over a separate stream so the program retains stdin.
    with tempfile.TemporaryDirectory(prefix='es40-', dir=env['TMPDIR']) as temp:
        archive = Path(temp) / 'run.tar.gz'
        with tarfile.open(archive, 'w:gz') as out:
            out.add(binary, arcname='program', recursive=False)
            for entry in json.loads(env.get('ALPHA_RUN_FILES', '[]')):
                path = Path(entry)
                if path.is_absolute() or '..' in path.parts:
                    fail('ALPHA_RUN_FILES entries must be relative paths within the working directory')
                if path.name == 'program':
                    fail('The runner reserves the filename program')
                if path.is_symlink() or any(item.is_symlink() for item in path.rglob('*')):
                    fail('Runner fixtures must contain regular files and directories, not symlinks')
                out.add(path, arcname=str(path))
        upload = 'umask 077; mkdir ' + shlex.quote(remote) + ' && tar -xzf - -C ' + shlex.quote(remote)
        names = set(filter(None, env.get('ALPHA_RUN_ENV', '').split(',')))
        names.update(k for k in env if k.startswith('RUST_TEST_') or k in ('RUST_BACKTRACE', 'RUST_LIB_BACKTRACE'))
        assignments = [key + '=' + env[key] for key in sorted(names) if key in env]
        run = ('trap ' + shlex.quote('rm -rf -- ' + shlex.quote(remote)) + ' EXIT HUP INT TERM; '
               'cd ' + shlex.quote(remote) + ' && ' + shlex.join(['env', *assignments, './program', *args[1:]]))
        with archive.open('rb') as stream:
            descriptor = stream.fileno()
            program_input = os.dup(sys.stdin.fileno())
            terminal = None
            try:
                command = (shlex.join([*ssh, upload]) + ' <&' + str(descriptor) + ' && '
                           + shlex.join([*ssh, run]) + ' <&' + str(program_input))
                invocation = ['su', local_user, '-s', '/bin/sh', '-c', command] if local_user else ['/bin/sh', '-c', command]
                if local_user:
                    # PAM must read authentication from the terminal, never from
                    # the application's piped input. SSH gets a separate fd.
                    terminal = open('/dev/tty', 'rb')
                result = subprocess.run(invocation, pass_fds=(descriptor, program_input),
                                        stdin=terminal, env=env)
            finally:
                os.close(program_input)
                if terminal:
                    terminal.close()
        raise SystemExit(result.returncode)


def cargo(args):
    if args and args[0] == 'alpha':
        args = args[1:]
    if not args or args[0] in ('--help', '-h', 'help'):
        print('''Usage: cargo alpha <build|check|run|test|new|init|fetch|update|metadata|clean> [Cargo options]
       cargo alpha doctor [--verify]
       cargo alpha run|test --runner qemu|es40 [Cargo options] [-- program arguments]

QEMU is the default runner. Configure ES40 with ALPHA_SSH_TARGET and optional
ALPHA_SSH_LOCAL_USER, or SDK config.json. Cargo options such as --release,
--offline, --locked, --manifest-path, --features and --package pass through.
Downloads are enabled by default; caches and temporary files stay in SDK var/.
Set ALPHA_CACHE_DIR to move caches. Set ALPHA_RUN_FILES to a JSON list of relative
fixture paths for ES40, and ALPHA_RUN_ENV to comma-separated environment names.
Unit/integration tests use aborting panics. Doctests are not enabled by default.''')
        return
    operation, *options = args
    if operation == 'doctor':
        doctor('--verify' in options)
        return
    if operation not in ('build', 'check', 'run', 'test', 'new', 'init', 'fetch', 'update', 'metadata', 'clean'):
        fail('Unsupported command ' + operation + '; see cargo alpha --help')
    env = environment()
    parsed = []
    i = 0
    while i < len(options):
        value = options[i]
        if value == '--':
            parsed.extend(options[i:])
            break
        if value == '--runner' or value.startswith('--runner='):
            if '=' in value:
                selected = value.split('=', 1)[1]
            else:
                i += 1
                if i == len(options):
                    fail('--runner needs qemu or es40')
                selected = options[i]
            if selected not in ('qemu', 'es40'):
                fail('--runner needs qemu or es40')
            env['ALPHA_RUNNER'] = selected
        elif value == '--target' or value.startswith('--target='):
            fail('cargo alpha selects its own Alpha target; omit --target')
        else:
            parsed.append(value)
        i += 1
    command = [ROOT / 'toolchain/bin/cargo', operation]
    if operation in ('fetch', 'update', 'metadata'):
        command += ['--config', 'patch.crates-io.libc.path=' + json.dumps(str(ROOT / 'libc'))]
    if operation in ('build', 'check', 'run', 'test'):
        flags = ['-Zcodegen-backend=' + str(ROOT / 'lib/librustc_codegen_gcc.so'),
                 '--sysroot=' + str(ROOT / 'toolchain'), '-C', 'linker=' + str(ROOT / 'bin/alpha-cc'),
                 '-C', 'panic=abort', '-Zpanic-abort-tests']
        extra = env.get('CARGO_ENCODED_RUSTFLAGS')
        flags.extend(extra.split('\x1f') if extra else shlex.split(env.get('RUSTFLAGS', '')))
        env['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join(flags)
        env.pop('RUSTFLAGS', None)
        command += ['--target', ROOT / 'targets' / (TARGET + '.json'),
                    '-Zbuild-std=std,panic_abort' + (',test' if operation == 'test' else ''),
                    '-Zbuild-std-features=compiler-builtins-no-f16-f128',
                    '--config', 'patch.crates-io.libc.path=' + json.dumps(str(ROOT / 'libc')),
                    '--config', 'target.' + TARGET + '.runner=' + json.dumps([str(ROOT / 'bin/alpha-runner')])]
        if operation == 'test':
            command += ['-Zpanic-abort-tests']
            env['ALPHA_TEST'] = '1'
            before_separator = parsed[:parsed.index('--')] if '--' in parsed else parsed
            if '--doc' in before_separator:
                fail('Doctests are not yet supported; use unit or integration tests')
            selectors = ('--lib', '--bins', '--bin', '--tests', '--test', '--examples', '--example', '--all-targets', '--benches', '--bench')
            if not any(a.split('=')[0] in selectors for a in before_separator):
                command += ['--tests']
    execute([*command, *parsed], env)


def main():
    if sys.version_info < (3, 11):
        fail('Python 3.11 or newer is required')
    name = Path(sys.argv[0]).name
    if name in ('alpha-rustc', 'alpha-rustdoc', 'alpha-cc', 'alpha-ar'):
        tool(name, sys.argv[1:])
    elif name == 'alpha-runner':
        runner(sys.argv[1:])
    else:
        cargo(sys.argv[1:])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError) as error:
        fail(str(error))
