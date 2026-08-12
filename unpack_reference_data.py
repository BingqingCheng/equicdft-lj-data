#!/usr/bin/env python3
"""Extract every compressed reference-data archive into ``data/``."""

from pathlib import Path
from zipfile import ZipFile


REPOSITORY = Path(__file__).resolve().parent
ARCHIVE_DIRECTORY = REPOSITORY / "reference_data"


def main():
    archives = sorted(ARCHIVE_DIRECTORY.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(
            "no ZIP archives found in {}".format(ARCHIVE_DIRECTORY)
        )
    for archive in archives:
        print("extracting {}".format(archive.name), flush=True)
        with ZipFile(archive) as handle:
            handle.extractall(REPOSITORY)
    print("reference data are available under {}".format(REPOSITORY / "data"))


if __name__ == "__main__":
    main()
