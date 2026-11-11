#!/usr/bin/env python3
"""Exercise the SDK from separate Cargo projects and retain regression logs."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--abi-only', action='store_true')
    parser.add_argument('--runner', choices=('qemu','es40'), default='qemu')
    args = parser.parse_args()
    sdk, work = args.sdk.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    logs = work/'logs'; logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['PATH'] = str(sdk/'bin') + os.pathsep + env['PATH']
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cargo = [str(sdk/'toolchain/bin/cargo'), 'alpha']
    results = {}

    def run(label, command, expected=0, extra=None, input_text=None):
        print(label, flush=True)
        with (logs/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=work,
                                    env=env | (extra or {}), stdout=log,
                                    stderr=None if args.runner == 'es40' else subprocess.STDOUT, input=input_text, text=True, timeout=600)
        results[label] = {'expected':expected, 'actual':result.returncode}
        (logs/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        if result.returncode != expected:
            raise SystemExit(f'{label}: exit {result.returncode}; see {logs/(label+".log")}')

    if not args.abi_only:
        hello = work/'hello'
        if not hello.exists():
            run('new-project', [*cargo, 'new','--vcs','none',hello])
        run('hello', [*cargo,'run','--manifest-path',hello/'Cargo.toml','--runner',args.runner])
        compat = work/'compat'
        shutil.copytree(SOURCE/'tests/compat', compat, dirs_exist_ok=True)
        for profile in ('debug','release'):
            flags = ['--release'] if profile == 'release' else []
            run('compat-'+profile, [*cargo,'test','--locked','--manifest-path',compat/'Cargo.toml','--runner',args.runner,*flags])
        run('check', [*cargo,'check','--locked','--manifest-path',compat/'Cargo.toml'])
        run('failure-status', [*cargo,'test','--offline','--manifest-path',compat/'Cargo.toml','--runner',args.runner,
                              '--lib','intentional_failure','--','--ignored'], expected=101)
        run('exit-status', [*cargo,'run','--offline','--manifest-path',compat/'Cargo.toml','--runner',args.runner,
                           '--','--exit-code','37'], expected=37)
        run('quoted-arguments', [*cargo,'run','--offline','--manifest-path',compat/'Cargo.toml','--runner',args.runner,
                                '--','--name',"Alpha 'quoted'; dollar $" ])
        assert "hello Alpha 'quoted'; dollar $" in (logs/'quoted-arguments.log').read_text()
        (work/'fixture spaced.txt').write_text('fixture payload\n')
        run('runner-io', [*cargo,'run','--offline','--manifest-path',compat/'Cargo.toml','--runner',args.runner,
                         '--','--read-stdin','--input','fixture spaced.txt'],
            input_text='stdin payload\n', extra={'ALPHA_RUN_FILES':json.dumps(['fixture spaced.txt']),
                       'ALPHA_RUN_ENV':'ALPHA_SDK_TEST_VALUE','ALPHA_SDK_TEST_VALUE':'environment payload'})
        output=(logs/'runner-io.log').read_text()
        assert 'stdin: stdin payload' in output and 'fixture: fixture payload' in output
        if args.runner == 'qemu':
            assert 'environment: environment payload' in output

    # These low-level probes provide their own _start and need no C startup files.
    no_std = {'RUSTFLAGS':'-C link-arg=-nostdlib -C link-arg=-nostartfiles -C link-arg=-lc'}
    runner = [sdk/'bin/alpha-runner']
    build_id=(sdk/'build-id').read_text().strip()
    target_dir = Path(env.get('CARGO_TARGET_DIR', Path(env.get('ALPHA_CACHE_DIR',sdk/'var'))/'target'/build_id))
    target = target_dir/'alphaev6-unknown-linux-gnu'
    for crate in ('abi_probe','alpha_test','libc_probe'):
        destination=work/crate
        shutil.copytree(REPO/crate, destination, dirs_exist_ok=True)
        manifest=destination/'Cargo.toml'
        manifest.write_text(manifest.read_text().replace('"../libc-alpha"',json.dumps(str(sdk/'libc'))))
    run('c-constants-build',[sdk/'bin/alpha-cc','-no-pie',work/'libc_probe/constants.c','-o',work/'c-constants'])
    run('c-constants',[*runner,work/'c-constants'])
    for profile in ('debug','release'):
        flags=['--release'] if profile=='release' else []
        for crate in ('abi_probe','alpha_test','libc_probe'):
            run(crate+'-'+profile+'-build', [*cargo,'build','--offline','--manifest-path',work/crate/'Cargo.toml',*flags],extra=no_std)
        for test in ('alpha_test','abi_scalars','abi_pair_arg','abi_pair_ret','abi_triple_arg','abi_triple_callback','abi_extended','layouts','runtime'):
            run(test+'-'+profile,[*runner,target/profile/test],extra={'ALPHA_RUNNER':args.runner})
        run('constants-'+profile,[*runner,target/profile/'constants'],extra={'ALPHA_RUNNER':args.runner})
        run('constants-'+profile+'-compare',['python3',work/'libc_probe/compare_constants.py',logs/'c-constants.log',logs/('constants-'+profile+'.log')])
    print('SDK regression checks: PASS')


if __name__ == '__main__':
    main()
