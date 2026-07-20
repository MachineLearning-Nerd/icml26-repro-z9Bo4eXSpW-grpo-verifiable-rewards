"""Create a portable archive of source, code, results, and local logbook evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
INCLUDE = ("README.md", "STATUS.md", "docs", "repro", "outputs", ".trackio")
EXCLUDE = {"reproduction_bundle.zip", "reproduction_bundle_manifest.json"}


def keep(path: Path) -> bool:
    return path.is_file() and path.name not in EXCLUDE and "__pycache__" not in path.parts and "local" not in path.parts


def main() -> None:
    paths: list[Path] = []
    for name in INCLUDE:
        candidate = ROOT / name
        if candidate.is_file() and keep(candidate):
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(sorted(path for path in candidate.rglob("*") if keep(path)))
    target = OUT / "reproduction_bundle.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    payload = {
        "file_count": len(paths),
        "size_bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (OUT / "reproduction_bundle_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
