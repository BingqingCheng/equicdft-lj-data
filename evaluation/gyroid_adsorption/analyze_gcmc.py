#!/usr/bin/env python3
"""Compare homogeneous and gyroid GCMC densities across chemical potential."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_number_series(path):
    """Read numeric rows from a LAMMPS ``fix ave/time`` output."""

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values = [float(value) for value in line.split()]
            if values[0] > 0.0:
                rows.append(values)
    if not rows:
        raise ValueError("no production samples in {}".format(path))
    return np.asarray(rows, dtype=float)


def block_sem(steps, values, block_steps=200000):
    """Estimate the standard error from consecutive fixed-length blocks."""

    block_ids = ((steps.astype(np.int64) - 1) // block_steps).astype(int)
    block_means = np.asarray(
        [values[block_ids == index].mean() for index in np.unique(block_ids)]
    )
    if len(block_means) < 2:
        return float("nan"), len(block_means)
    return float(block_means.std(ddof=1) / math.sqrt(len(block_means))), len(
        block_means
    )


def load_campaign(campaign):
    """Return one analyzed record per completed state point."""

    records = []
    for summary_path in sorted(campaign.glob("mu-*/run-001/gcmc_summary.json")):
        run_directory = summary_path.parent
        summary = json.loads(summary_path.read_text())
        statepoint = json.loads((run_directory / "statepoint.json").read_text())
        series = read_number_series(run_directory / "particle_number_timeseries.dat")
        volume = float(statepoint["box_length"]) ** 3

        number_sem, n_blocks = block_sem(series[:, 0], series[:, 1])
        record = {
            "mu": float(statepoint["chemical_potential"]),
            "temperature": float(statepoint["temperature"]),
            "mean_number": float(summary["mean_particle_number"]),
            "density": float(summary["mean_box_density"]),
            "density_sem": number_sem / volume,
            "number_variance": float(summary["particle_number_variance"]),
            "S0": float(summary["S0_from_number_fluctuations"]),
            "n_blocks": n_blocks,
        }

        if statepoint["system"] == "gyroid":
            # The nominal pore is g(r)>0. Gyroid antisymmetry makes its volume
            # exactly half of the periodic box for the g0=0 surface.
            pore_volume = 0.5 * volume
            pore_number_sem, pore_blocks = block_sem(
                series[:, 0], series[:, 2]
            )
            record.update(
                {
                    "pore_number": float(summary["mean_pore_particle_number"]),
                    "pore_density": float(summary["mean_pore_particle_number"])
                    / pore_volume,
                    "pore_density_sem": pore_number_sem / pore_volume,
                    "pore_blocks": pore_blocks,
                }
            )
        records.append(record)
    return sorted(records, key=lambda record: record["mu"])


def write_table(path, bulk, gyroid):
    if [row["mu"] for row in bulk] != [row["mu"] for row in gyroid]:
        raise ValueError("bulk and gyroid chemical-potential grids differ")

    columns = [
        "mu",
        "temperature",
        "rho_bulk",
        "rho_bulk_sem",
        "S0_bulk",
        "rho_gyroid_box",
        "rho_gyroid_box_sem",
        "rho_gyroid_pore",
        "rho_gyroid_pore_sem",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for bulk_row, gyroid_row in zip(bulk, gyroid):
            writer.writerow(
                {
                    "mu": bulk_row["mu"],
                    "temperature": bulk_row["temperature"],
                    "rho_bulk": bulk_row["density"],
                    "rho_bulk_sem": bulk_row["density_sem"],
                    "S0_bulk": bulk_row["S0"],
                    "rho_gyroid_box": gyroid_row["density"],
                    "rho_gyroid_box_sem": gyroid_row["density_sem"],
                    "rho_gyroid_pore": gyroid_row["pore_density"],
                    "rho_gyroid_pore_sem": gyroid_row["pore_density_sem"],
                }
            )


def plot_isotherm(path, bulk, gyroid):
    mu = np.asarray([row["mu"] for row in bulk])
    rho_bulk = np.asarray([row["density"] for row in bulk])
    err_bulk = np.asarray([row["density_sem"] for row in bulk])
    rho_gyroid = np.asarray([row["density"] for row in gyroid])
    err_gyroid = np.asarray([row["density_sem"] for row in gyroid])

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    ax.errorbar(
        mu,
        rho_bulk,
        yerr=err_bulk,
        color="#1f4e79",
        marker="o",
        markersize=5.5,
        linewidth=1.7,
        capsize=2.5,
        label=r"homogeneous bulk, $L=8\sigma$",
    )
    ax.errorbar(
        mu,
        rho_gyroid,
        yerr=err_gyroid,
        color="#c45a18",
        marker="^",
        markersize=6.0,
        linewidth=1.7,
        capsize=2.5,
        label=r"gyroid box average, $L=16\sigma$",
    )
    ax.set_xlabel(r"LAMMPS chemical potential, $\mu/\epsilon$")
    ax.set_ylabel(r"number density, $\rho\sigma^3$")
    ax.set_xlim(mu.min() - 0.12, mu.max() + 0.12)
    ax.set_ylim(bottom=0.0)
    ax.text(
        0.03,
        0.96,
        r"$T^*=1.2$",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--bulk",
        type=Path,
        default=root / "data" / "gyroid_adsorption" / "bulk_gcmc",
    )
    parser.add_argument(
        "--gyroid",
        type=Path,
        default=root / "data" / "gyroid_adsorption" / "gyroid_gcmc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "gyroid_adsorption" / "gcmc_mu_density_T1.2.png",
    )
    args = parser.parse_args()

    bulk = load_campaign(args.bulk)
    gyroid = load_campaign(args.gyroid)
    if len(bulk) != 11 or len(gyroid) != 11:
        raise ValueError("expected 11 completed state points in each campaign")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_path = args.output.with_suffix(".csv")
    write_table(table_path, bulk, gyroid)
    plot_isotherm(args.output, bulk, gyroid)
    print("Wrote {} and {}.".format(args.output, table_path))


if __name__ == "__main__":
    main()
