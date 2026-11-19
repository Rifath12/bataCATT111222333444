from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import hashlib, json, shutil, time

ROOT=Path(__file__).resolve().parent

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def download(entry):
    path=ROOT/entry['file']; path.parent.mkdir(exist_ok=True,parents=True)
    if path.is_file() and digest(path)==entry['sha256']:
        return
    for attempt in range(3):
        try:
            with urlopen(entry['url'],timeout=100) as source, path.with_suffix('.partial').open('wb') as target:
                shutil.copyfileobj(source,target,1024*1024)
            if digest(path.with_suffix('.partial'))!=entry['sha256']:
                raise ValueError('Checksum mismatch for '+str(path))
            path.with_suffix('.partial').replace(path);print(path.name,flush=True);return
        except OSError:
            if attempt==2:raise
            time.sleep(2)

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download,json.loads((ROOT/'sources.json').read_text())['sources']))
