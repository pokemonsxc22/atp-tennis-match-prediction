"""Download a pinned educational-use data snapshot with provenance."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
def fetch(url):
    return urlopen(Request(url, headers={'User-Agent': 'Tennis-Research-Project'}), timeout=60).read()

commit = json.loads(fetch('https://api.github.com/repos/Tennismylife/TML-Database/commits/master'))['sha']
manifest = {'repository': 'https://github.com/Tennismylife/TML-Database', 'commit': commit, 'files': []}
for year in range(2014, 2026):
    url = f'https://raw.githubusercontent.com/Tennismylife/TML-Database/{commit}/{year}.csv'
    content = fetch(url)
    (DATA / f'{year}.csv').write_bytes(content)
    manifest['files'].append({'file': f'{year}.csv', 'url': url, 'sha256': hashlib.sha256(content).hexdigest()})
    print(year, len(content), flush=True)
(DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2))
(DATA / 'SOURCE_README.md').write_bytes(fetch(f'https://raw.githubusercontent.com/Tennismylife/TML-Database/{commit}/README.md'))
