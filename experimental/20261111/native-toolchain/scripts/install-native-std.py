from pathlib import Path
import hashlib,json,shutil
w=Path(__file__).resolve().parent
build=w/'rust-build'
triple='alphaev6-unknown-linux-gnu'
stamp=build/'x86_64-unknown-linux-gnu/stage0-std'/triple/'dist/.libstd-stamp'
dst=build/triple/'stage1/lib/rustlib'/triple/'lib'
dst.mkdir(parents=True,exist_ok=True)
records=[]
for item in stamp.read_bytes().split(b'\0'):
    if not item:continue
    kind=chr(item[0]);src=Path(item[1:].decode())
    if kind=='h':continue
    assert kind in ('t','s'),kind
    assert src.is_relative_to(build),src
    target=(dst/'self-contained' if kind=='s' else dst)/src.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,target)
    records.append({'source':str(src),'installed':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
assert any(Path(r['installed']).name.startswith('libstd-') for r in records)
(w/'native-std-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(f'Installed {len(records)} Alpha standard-library artifacts from bootstrap stamp')
