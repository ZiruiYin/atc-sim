"""Regenerate env_manifest.json after adding or removing files in environment/.

The repo root is also the GitHub Pages root. The browser (PyodideEngine) fetches
every file listed in env_manifest.json at boot and mounts them into Pyodide's
virtual filesystem. So this list must stay in sync with what's under
environment/.

Run after adding/removing/renaming files in environment/:

    python build_pages.py

Then commit env_manifest.json. Both the SIMULATED ('test') and EGLL airports
ship. Only files under environment/ are listed — the AUTO planner (auto_plan/,
incl. best.pt + torch) is backend-only and intentionally absent from the
Pyodide build.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_ENV = ROOT / 'environment'
MANIFEST = ROOT / 'env_manifest.json'
NOJEKYLL = ROOT / '.nojekyll'

EXCLUDE_DIRS = {'__pycache__'}
INCLUDE_EXTS = {'.py', '.json'}
# Both airports ship now (SIMULATED + EGLL); nothing under environment/ is
# excluded by name.
EXCLUDE_FILE_PREFIXES = ()


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
    if not SRC_ENV.exists():
        print(f"missing {SRC_ENV}", file=sys.stderr)
        sys.exit(1)

    files = collect_env_files()
    MANIFEST.write_text(json.dumps(files, indent=2) + '\n', encoding='utf-8')

    if not NOJEKYLL.exists():
        NOJEKYLL.write_text('', encoding='utf-8')

    print(f"env_manifest.json updated: {len(files)} entries")


if __name__ == '__main__':
    main()
