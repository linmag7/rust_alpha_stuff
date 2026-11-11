#!/usr/bin/env python3
"""Run over SSH on the ES40; emit only development headers and glibc files."""
import glob
import os
import subprocess
import sys
version = subprocess.check_output(['getconf', 'GNU_LIBC_VERSION'], text=True).strip()
if version != 'glibc 2.41':
    sys.exit('Expected glibc 2.41, found ' + version)
files = ['usr/include']
for pattern in ('*crt*.o', 'ld-linux.so*', 'libc.*', 'libc_nonshared.a', 'libm.*',
                'libmvec.*', 'libpthread.*', 'libdl.*', 'librt.*', 'libresolv.*',
                'libutil.*', 'libanl.*', 'libthread_db.*', 'libBrokenLocale.*',
                'libnss_files.*', 'libnss_dns.*', 'libnss_compat.*'):
    files.extend(p.lstrip('/') for p in glob.glob('/usr/lib/' + pattern))
if 'usr/lib/libc.so.6.1' not in files or 'usr/lib/crt1.o' not in files:
    sys.exit('Missing Alpha glibc development files')
os.execvp('tar', ['tar', '-C', '/', '-czhf', '-', *sorted(set(files))])
