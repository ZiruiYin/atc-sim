"""Build the GitHub Pages deployment into docs/.

Run before pushing to refresh the static site:

    python build_pages.py

Produces:
    docs/index.html               <- copy of static/index.html
    docs/environment/...          <- copy of environment/ (.py + data/*.json)
    docs/env_manifest.json        <- list of files the PyodideEngine fetches
    docs/.nojekyll                <- disable Jekyll on GitHub Pages

Configure GitHub Pages to serve from /docs on the main branch.
"""

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_HTML = ROOT / 'static' / 'index.html'
SRC_ENV = ROOT / 'environment'
OUT = ROOT / 'docs'

EXCLUDE_DIRS = {'__pycache__'}
INCLUDE_EXTS = {'.py', '.json'}
# EGLL data is kept in the repo as legacy reference but excluded from the
# GitHub Pages deployment — only the 'test' airport ships.
EXCLUDE_FILE_PREFIXES = ('egll',)


def _retry_rm(path):
    """Remove a file or empty dir with retries (OneDrive holds locks briefly)."""
    for delay in (0, 0.2, 0.5, 1.0, 2.0):
        if delay:
            time.sleep(delay)
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                path.rmdir()
            return
        except PermissionError:
            continue
        except FileNotFoundError:
            return
    # last shot — let it raise
    if path.is_file():
        path.unlink()
    else:
        path.rmdir()


def clean_output_dir(d):
    if not d.exists():
        return
    # Files first (deepest first), then empty dirs bottom-up.
    for path in sorted(d.rglob('*'), key=lambda p: -len(p.parts)):
        if path.is_file():
            _retry_rm(path)
    for path in sorted(d.rglob('*'), key=lambda p: -len(p.parts)):
        if path.is_dir():
            _retry_rm(path)
    _retry_rm(d)


def collect_env_files():
    rels = []
    for path in SRC_ENV.rglob('*'):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix not in INCLUDE_EXTS:
            continue
        if path.name.startswith(EXCLUDE_FILE_PREFIXES):
            continue
        rel = path.relative_to(ROOT).as_posix()
        rels.append(rel)
    rels.sort()
    return rels


def main():
    if not SRC_HTML.exists():
        print(f"missing {SRC_HTML}", file=sys.stderr)
        sys.exit(1)
    if not SRC_ENV.exists():
        print(f"missing {SRC_ENV}", file=sys.stderr)
        sys.exit(1)

    # Overwrite-in-place. Avoiding rmtree because OneDrive briefly locks
    # directories under sync, which makes a full clean unreliable on Windows.
    # If you need a clean build, delete docs/ manually first.
    OUT.mkdir(exist_ok=True)

    shutil.copy2(SRC_HTML, OUT / 'index.html')

    env_files = collect_env_files()
    for rel in env_files:
        src = ROOT / rel
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    (OUT / 'env_manifest.json').write_text(
        json.dumps(env_files, indent=2) + '\n',
        encoding='utf-8',
    )

    (OUT / '.nojekyll').write_text('', encoding='utf-8')

    print(f"built docs/ with {len(env_files)} env files")


if __name__ == '__main__':
    main()
