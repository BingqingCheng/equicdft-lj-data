#!/usr/bin/env python3
"""Summarize the direct fixed-separation colloid force measurements."""

import argparse
import csv
import json
import math
from pathlib import Path


def summarize_force(path, block_samples):
    """Return the production mean and a block estimate of its standard error."""

    count = 0
    total = 0.0
    total_squared = 0.0
    block_total = 0.0
    block_count = 0
    block_means = []
    last_step = None

    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 4:
                raise ValueError("invalid force row in {}: {}".format(path, line))
            step = int(fields[0])
            last_step = step
            force = float(fields[3])
            if not math.isfinite(force):
                raise ValueError("non-finite force in {}".format(path))

            # The file contains both endpoints, step 0 and step 4,000,000.
            # Exclude the initial endpoint so the 400,000 production intervals
            # divide exactly into equal-duration statistical blocks.
            if step == 0:
                continue

            count += 1
            total += force
            total_squared += force * force
            block_total += force
            block_count += 1
            if block_count == block_samples:
                block_means.append(block_total / block_count)
                block_total = 0.0
                block_count = 0

    if count < 2 or last_step != 4000000:
        raise ValueError("incomplete production force record: {}".format(path))
    if block_count:
        raise ValueError(
            "force record is not divisible into complete blocks: {}".format(path)
        )
    if len(block_means) < 2:
        raise ValueError("at least two force blocks are required")

    mean = total / count
    variance = (total_squared - count * mean * mean) / (count - 1)
    block_mean = sum(block_means) / len(block_means)
    block_variance = sum(
        (value - block_mean) ** 2 for value in block_means
    ) / (len(block_means) - 1)

    return {
        "mean_force": mean,
        "instantaneous_std": math.sqrt(max(variance, 0.0)),
        "block_sem": math.sqrt(max(block_variance, 0.0) / len(block_means)),
        "n_samples": count,
        "n_blocks": len(block_means),
        "last_step": last_step,
    }


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-samples", type=int, default=4000)
    parser.add_argument("--timestep", type=float, default=0.005)
    parser.add_argument(
        "--root",
        type=Path,
        default=repository / "data" / "colloid_bridge" / "reference_scan",
        help="directory containing the D* run folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    arguments = parser.parse_args()
    if arguments.block_samples <= 0:
        raise ValueError("block-samples must be positive")

    root = arguments.root.expanduser().resolve()
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else root / "mean_force_summary.csv"
    )
    rows = []
    for run_directory in sorted(root.glob("D*"), key=lambda path: float(path.name[1:])):
        statepoint = json.loads((run_directory / "statepoint.json").read_text())
        summary = summarize_force(
            run_directory / "colloid_force_timeseries.dat",
            arguments.block_samples,
        )
        sample_every = int(statepoint["force_sample_every"])
        summary.update(
            {
                "separation": float(statepoint["separation"]),
                "block_samples": arguments.block_samples,
                "block_duration": (
                    arguments.block_samples * sample_every * arguments.timestep
                ),
            }
        )
        rows.append(summary)

    fieldnames = (
        "separation",
        "mean_force",
        "block_sem",
        "instantaneous_std",
        "n_samples",
        "block_samples",
        "block_duration",
        "n_blocks",
        "last_step",
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            "D={:.1f}: F_D={:+.6f} +/- {:.6f}".format(
                row["separation"], row["mean_force"], row["block_sem"]
            )
        )
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
