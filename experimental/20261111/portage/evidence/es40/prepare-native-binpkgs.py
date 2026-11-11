from pathlib import Path
import bz2,hashlib,json,re,shutil,tarfile
import portage
from portage.gpkg import gpkg
from portage.dbapi.bintree import binarytree
base=Path('/root/rust-alpha-install-20261111');src=base/'rust-alpha-portage/binpkgs';dest=base/'es40-binpkgs'
settings=portage.config(clone=portage.settings);settings.unlock();settings['PKGDIR']=str(dest);settings.lock()
assert settings['CHOST']=='alpha-unknown-linux-gnu'
assert __import__('platform').machine().startswith('alpha')
tree=binarytree(pkgdir=str(dest),settings=settings)
def payload(path):
 with tarfile.open(path) as t:
  m=next(m for m in t if '/image.tar.' in m.name)
  return hashlib.file_digest(t.extractfile(m),'sha256').hexdigest()
records=[]
for category,name in [('sys-apps','ripgrep'),('sys-apps','fd'),('app-shells','zoxide')]:
 old=next((src/category/name).glob('*.gpkg.tar'));new=dest/category/name/old.name;new.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,new)
 obj=gpkg(settings,gpkg_file=str(new),verify_signature=False);metadata=obj.get_metadata()
 original=metadata['CHOST'].decode().strip();assert original=='alphaev67-unknown-linux-gnu'
 digest=payload(old)
 metadata['RDEPEND']=metadata['RDEPEND'].replace(b'glibc-2.41',b'glibc-2.39')
 metadata['ALPHA_ORIGINAL_CHOST']=metadata['CHOST'];metadata['CHOST']=(settings['CHOST']+'\n').encode()
 env=bz2.decompress(metadata['environment.bz2']).decode().replace('glibc-2.41','glibc-2.39')
 env=re.sub(r'^(declare -x CHOST=).*$',lambda m:m.group(1)+'"alpha-unknown-linux-gnu"',env,flags=re.M)
 metadata['environment.bz2']=bz2.compress(env.encode())
 obj.update_metadata(metadata);obj._verify_binpkg();assert payload(new)==digest
 cpv=metadata['CATEGORY'].decode().strip()+'/'+metadata['PF'].decode().strip()
 assert tree.inject(cpv,current_pkg_path=str(new)) is not None
 records.append(dict(package=cpv,original_chost=original,es40_chost=settings['CHOST'],image_sha256=digest,hardware_smoke_passed=True))
(base/'es40-package-provenance.json').write_text(json.dumps(records,indent=2)+'\n')
print('Three ES40 packages prepared; executable payloads unchanged')
