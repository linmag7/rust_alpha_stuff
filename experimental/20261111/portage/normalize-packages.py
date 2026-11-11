#!/usr/bin/env python3
"""Normalize ownership for these root-owned packages built by an unprivileged host.

Use Portage's GPKG writer and index updater; do not edit checksums by hand.
This host-only step is unnecessary when Portage builds as root on Gentoo.
"""
from pathlib import Path
import os,sys,tarfile,bz2,re,subprocess
here=Path(__file__).resolve().parent
project=here.parents[3]
w=project/'work/portage-20261111'
os.environ.update(PORTAGE_OVERRIDE_EPREFIX='',PORTAGE_CONFIGROOT=str(w/'config'),ROOT=str(w/'root'),TMPDIR=str(w/'tmp'))
import portage
from portage.gpkg import gpkg
settings=portage.settings
settings.unlock()
settings['BINPKG_COMPRESS_FLAGS_XZ']='-T4 -3'
settings.lock()
bintree=portage.db[settings['ROOT']]['bintree']
cpv=sys.argv[1]
category,pf=cpv.split('/')
image=w/'portage-tmp/portage'/cpv/'image'
assert image.is_dir()
original=max((p for p in (w/'binpkgs'/category).rglob('*.gpkg.tar') if p.name.startswith(pf+'-')),key=lambda p:p.stat().st_mtime)
metadata=gpkg(settings,gpkg_file=str(original),verify_signature=False).get_metadata()
if category != 'dev-lang':
 # Refresh the packaging-only setup hook from the final eclass without rebuilding code.
 function=subprocess.check_output(['bash','-c','EAPI=8; EXPORT_FUNCTIONS() { :; }; source "$1"; declare -f alpha-cargo_pkg_setup','bash',str(here/'overlay/eclass/alpha-cargo.eclass')],text=True)
 saved=bz2.decompress(metadata['environment.bz2']).decode()
 saved,count=re.subn(r'alpha-cargo_pkg_setup \(\)\s*\{.*?^}',lambda m:function.rstrip(),saved,count=1,flags=re.S|re.M)
 assert count==1
 metadata['environment.bz2']=bz2.compress(saved.encode())
output=w/'tmp'/(pf+'-root-owned.gpkg.tar')
original_addfile=tarfile.TarFile.addfile
def root_owned(self,member,fileobj=None):
 member.uid=member.gid=0
 member.uname=member.gname='root'
 return original_addfile(self,member,fileobj)
tarfile.TarFile.addfile=root_owned
try:
 gpkg(settings,basename=pf,gpkg_file=str(output),verify_signature=False).compress(str(image),metadata)
finally:
 tarfile.TarFile.addfile=original_addfile
gpkg(settings,gpkg_file=str(output),verify_signature=False)._verify_binpkg()
assert bintree.inject(cpv,current_pkg_path=str(output)) is not None
print(cpv,'root-owned GPKG generated and verified',flush=True)
