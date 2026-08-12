#!/usr/bin/env python3
"""Create a reproducible file-stratified split for comprehensive NVT data."""

import argparse
import json
import random
import re
from pathlib import Path


HELD_OUT_TEMPERATURES = {0.7, 1.1, 1.5}
TEMPERATURE_PATTERN = re.compile(r"nvt-T([0-9.]+)")


def inspect_extxyz(path):
    """Count complete frames and validate the common L=8, spacing=0.5 grid."""

    frames = 0
    with path.open() as handle:
        while True:
            first_line = handle.readline()
            if not first_line:
                break
            if not first_line.strip():
                continue
            n_grid = int(first_line)
            metadata = handle.readline()
            if not metadata:
                raise ValueError("Missing metadata line in {}".format(path))
            if n_grid != 4096:
                raise ValueError("Expected 4096 grid points in {}".format(path))
            if 'physical_lattice="8 0 0 0 8 0 0 0 8"' not in metadata:
                raise ValueError("Expected an L=8 physical cell in {}".format(path))
            if 'grid_spacing="0.5 0.5 0.5"' not in metadata:
                raise ValueError("Expected grid spacing 0.5 in {}".format(path))
            for _ in range(n_grid):
                if not handle.readline():
                    raise ValueError("Incomplete frame in {}".format(path))
            frames += 1
    return frames


def temperature_from_name(path):
    match = TEMPERATURE_PATTERN.search(path.name)
    if match is None:
        raise ValueError("Cannot parse temperature from {}".format(path))
    return float(match.group(1))


def family_name(collection, path):
    if collection == "general":
        return "general_3D"
    if collection == "dimensional":
        if "-1D-" in path.name:
            return "1D"
        if "-2D-" in path.name:
            return "2D"
    if collection == "mixed":
        return "mixed"
    if collection == "extended":
        for family in ("1D", "2D", "3D", "mixed"):
            if "-{}-".format(family) in path.name:
                return family
    raise ValueError("Cannot determine field family for {}".format(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-directory", type=Path, required=True)
    parser.add_argument("--dimensional-directory", type=Path, required=True)
    parser.add_argument("--mixed-directory", type=Path, required=True)
    parser.add_argument("--extended-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    directories = {
        "general": arguments.general_directory,
        "dimensional": arguments.dimensional_directory,
        "mixed": arguments.mixed_directory,
        "extended": arguments.extended_directory,
    }
    records = []
    totals = {"train": 0, "valid": 0, "test": 0}
    for collection, directory in directories.items():
        paths = sorted(directory.expanduser().resolve().glob("*.extxyz"))
        if not paths:
            raise FileNotFoundError("No top-level extxyz files in {}".format(directory))
        for path in paths:
            temperature = temperature_from_name(path)
            frame_count = inspect_extxyz(path)
            record = {
                "collection": collection,
                "filename": path.name,
                "temperature": temperature,
                "field_family": family_name(collection, path),
                "frames": frame_count,
            }
            if temperature in HELD_OUT_TEMPERATURES:
                record["partition"] = "test"
                record["valid_frame_indices"] = []
                totals["test"] += frame_count
            else:
                n_valid = max(1, int(round(arguments.valid_fraction * frame_count)))
                indices = list(range(frame_count))
                random.Random(
                    "{}:{}:{}".format(arguments.seed, collection, path.name)
                ).shuffle(indices)
                valid_indices = sorted(indices[:n_valid])
                record["partition"] = "development"
                record["valid_frame_indices"] = valid_indices
                totals["valid"] += n_valid
                totals["train"] += frame_count - n_valid
            records.append(record)

    manifest = {
        "name": "comprehensive_nvt_L8_grid0.5",
        "held_out_temperatures": sorted(HELD_OUT_TEMPERATURES),
        "valid_fraction_within_source_file": arguments.valid_fraction,
        "seed": arguments.seed,
        "totals": totals,
        "sources": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
