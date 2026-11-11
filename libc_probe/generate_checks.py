import json, pathlib, re
repo=pathlib.Path(__file__).resolve().parent.parent
work=repo.parent/'work'
values=json.loads((work/'alpha-c-constants.json').read_text())
skipfile=repo/'libc_probe/unavailable-constants.json'
skip=set(json.loads(skipfile.read_text())) if skipfile.exists() else set()
lines=['#![allow(deprecated)]', '#![no_std]', '#![no_main]', 'use alpha_libc_probe::*;', '#[unsafe(no_mangle)]', 'pub extern "C" fn _start()->! {']
for n in sorted(values.keys()-skip):
    lines += [f'    print(b"{n} "); number(libc::{n} as u64); print(b"\\n");']
lines+=['    finish(0)', '}']
(repo/'libc_probe/src/bin/constants.rs').write_text('\n'.join(lines)+'\n')
