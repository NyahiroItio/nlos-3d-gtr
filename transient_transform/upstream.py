"""Fetch and verify the unchanged code/data archive linked by the LCT authors.

The archive and extracted upstream files are local research inputs under the
ignored results directory. They are not redistributed as this repository's MIT
software. In particular, the 2018 archive contains no separate license file.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


SOURCE_PAGE = 'https://davidlindell.com/publications/nlos-lct'
FILE_ID = '1OoZ4JfkXY0bIGlb4dT22YjhhwplZjQOc'
ARCHIVE_URL = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
ARCHIVE_SHA256 = 'db3fef096e242baa642595cfd639da23f1f220e458d12c17832e4257a22024ec'
SOURCE_SHA256 = '824275cb1fbb5c33751781b7b7498d7c793a92dc07a456670ec169e38ea6b1f6'
DEFAULT_ROOT = Path(__file__).resolve().parent/'results'/'upstream'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_source(directory):
    path = Path(directory)/'cnlos_reconstruction.m'
    if not path.is_file() or sha256(path) != SOURCE_SHA256:
        raise ValueError('Original MATLAB source is missing or differs from the pinned author archive.')
    return path


def fetch(root=DEFAULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root/'cnlos_code.zip'
    if not archive.exists():
        import requests
        session = requests.Session()
        response = session.get(ARCHIVE_URL, timeout=(20, 180))
        response.raise_for_status()
        if response.content[:2] != b'PK':
            fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', response.text))
            if fields.get('id') != FILE_ID or fields.get('confirm') != 't':
                raise RuntimeError('Unexpected public Google Drive download response.')
            response = session.get('https://drive.usercontent.google.com/download',
                                   params=fields, timeout=(20, 180))
            response.raise_for_status()
        if response.content[:2] != b'PK':
            raise RuntimeError('The server did not return a ZIP archive.')
        if hashlib.sha256(response.content).hexdigest() != ARCHIVE_SHA256:
            raise ValueError('Archive hash changed; do not silently replace the reference baseline.')
        archive.write_bytes(response.content)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Cached archive does not match the pinned checksum.')
    target = root/'cnlos2018'
    target.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            relative = PurePosixPath(item.filename)
            if (len(relative.parts) == 2 and relative.parts[0] == 'confocal_nlos_code'
                    and relative.suffix in ('.m', '.mat') and '..' not in relative.parts):
                content = package.read(item)
                destination = target/relative.name
                destination.write_bytes(content)
                manifest[relative.name] = {'bytes': len(content),
                                           'sha256': hashlib.sha256(content).hexdigest()}
    verify_source(target)
    (root/'manifest.json').write_text(json.dumps({
        'source_page': SOURCE_PAGE, 'archive_url': ARCHIVE_URL,
        'archive_sha256': ARCHIVE_SHA256, 'files': manifest}, indent=2), encoding='utf-8')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(fetch(args.root))


if __name__ == '__main__':
    main()
