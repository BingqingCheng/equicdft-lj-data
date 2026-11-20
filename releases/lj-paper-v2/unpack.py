"""Extract downloaded release assets into data/study, preserving the frozen layout."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    destination = args.directory.resolve()
    assets = json.loads((ROOT / "manifests/release_assets.json").read_text())["assets"]
    for asset in assets:
        path = ROOT / asset["path"]
        if digest(path) != asset["sha256"]:
            raise ValueError("Archive checksum mismatch: " + str(path))
        archive = zipfile.ZipFile(path) if path.suffix == ".zip" else tarfile.open(path)
        with archive:
            entries = archive.infolist() if isinstance(archive, zipfile.ZipFile) else archive.getmembers()
            for entry in entries:
                name = entry.filename if isinstance(entry, zipfile.ZipInfo) else entry.name
                (destination / name).resolve().relative_to(destination)
                if isinstance(entry, tarfile.TarInfo) and not (entry.isfile() or entry.isdir()):
                    raise ValueError("Unexpected archive link: " + name)
                if (destination / name).exists() and not (destination / name).is_dir():
                    raise FileExistsError(destination / name)
            archive.extractall(destination)
        print("Extracted", path.name)


if __name__ == "__main__":
    main()
