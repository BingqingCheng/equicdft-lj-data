"""Check release files and downloaded assets without loading any model."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="Do not require downloaded release assets")
    args = parser.parse_args()
    checked = 0
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha(ROOT / name) != expected:
            raise ValueError("Checksum mismatch: " + name)
        checked += 1
    if not args.metadata_only:
        for asset in json.loads((ROOT / "manifests/release_assets.json").read_text())["assets"]:
            path = ROOT / asset["path"]
            if path.stat().st_size != asset["bytes"] or sha(path) != asset["sha256"]:
                raise ValueError("Asset mismatch: " + str(path))
            checked += 1
    print("PASS:", checked, "files checked")


if __name__ == "__main__":
    main()
