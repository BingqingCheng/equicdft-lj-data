#!/usr/bin/env python3
"""Plot MD and learned-cDFT liquid-vapor interface profiles."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from bulk_response_style import temperature_color


TEMPERATURES = (0.65, 0.80, 0.95, 1.05)
TEMPERATURE_COLORS = tuple(temperature_color(value) for value in TEMPERATURES)


def parse_arguments():
    script = Path(__file__).resolve()
    project = script.parents[2]
    default_fit = project / "data" / "bulk_response"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, default=default_fit)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=(
            project / "source_data" / "fig02_bulk_response" / "interface_snapshot_T0p95.extxyz"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project / "output" / "bulk_response",
    )
    return parser.parse_args()


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def temperature_label(temperature):
    return "{:.2f}".format(temperature).rstrip("0").rstrip(".")


def planar_profile(atoms, key):
    """Average one grid field over the two interface-parallel directions."""

    z = np.asarray(atoms.arrays["grid_center"][:, 2], dtype=float)
    values = np.asarray(atoms.arrays[key], dtype=float)
    unique_z, inverse = np.unique(np.round(z, decimals=12), return_inverse=True)
    counts = np.bincount(inverse)
    average = np.bincount(inverse, weights=values) / counts
    return unique_z, average


def metrics(reference, prediction):
    error = np.asarray(prediction) - np.asarray(reference)
    rmse = float(np.sqrt(np.mean(error**2)))
    centered = np.asarray(reference) - float(np.mean(reference))
    denominator = float(np.sum(centered**2))
    r2 = None if denominator == 0.0 else 1.0 - float(np.sum(error**2)) / denominator
    return rmse, r2


def load_profiles(fit_directory):
    profiles = {}
    summary = []
    for temperature in TEMPERATURES:
        label = temperature_label(temperature)
        path = (
            fit_directory
            / "slab-solves"
            / "nvt-slab-L6x6x24-T{}-solved.extxyz".format(label)
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        atoms = read(path)
        z, reference = planar_profile(atoms, "reference_density")
        _z, prediction = planar_profile(atoms, "density")
        rmse, r2 = metrics(reference, prediction)
        profiles[temperature] = {
            "z": z,
            "md": reference,
            "cdft": prediction,
        }
        summary.append(
            {
                "temperature": temperature,
                "profile_rmse": rmse,
                "profile_r2": r2,
                "converged": bool(atoms.info.get("solve_converged", False)),
                "iterations": int(atoms.info.get("solve_iterations", 0)),
                "source": str(path),
            }
        )
    return profiles, summary


def write_curve_table(path, profiles):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["temperature", "source", "z", "density"])
        for temperature in TEMPERATURES:
            profile = profiles[temperature]
            for z, density in zip(profile["z"], profile["md"]):
                writer.writerow([temperature, "MD", z, density])
            for z, density in zip(profile["z"], profile["cdft"]):
                writer.writerow([temperature, "cDFT", z, density])


def plot_panel(axis, profiles):
    """Draw the profiles and return handles for the temperature legend."""

    for temperature, color in zip(TEMPERATURES, TEMPERATURE_COLORS):
        profile = profiles[temperature]
        axis.plot(profile["z"], profile["md"], color=color, lw=1.35)
        axis.plot(
            profile["z"],
            profile["cdft"],
            color=color,
            lw=1.0,
            ls="--",
        )

    method_handles = [
        mlines.Line2D([], [], color="0.2", lw=1.35, label="MD"),
        mlines.Line2D([], [], color="0.2", lw=1.0, ls="--", label="cDFT"),
    ]
    method_legend = axis.legend(
        handles=method_handles,
        loc="upper left",
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.3,
    )
    axis.add_artist(method_legend)

    temperature_handles = [
        mlines.Line2D(
            [], [], color=color, lw=1.6, label=r"$T={:.2f}$".format(temperature)
        )
        for temperature, color in zip(TEMPERATURES, TEMPERATURE_COLORS)
    ]
    axis.set_xlim(0.0, 24.0)
    axis.set_ylim(0.0, 0.86)
    axis.set_xlabel(r"$z/\sigma$")
    axis.set_ylabel(r"$\rho\sigma^3$")
    return temperature_handles


def plot_snapshot(axis, path):
    """Render the genuine T=0.95 atomistic frame as vector markers."""

    atoms = read(path)
    positions = np.asarray(atoms.positions, dtype=float)
    order = np.argsort(positions[:, 1])
    axis.scatter(
        positions[order, 2],
        positions[order, 0],
        s=5.5,
        color="#1729D4",
        edgecolor="#07168F",
        linewidth=0.18,
        alpha=0.96,
        clip_on=True,
    )
    axis.set_xlim(0.0, 24.0)
    axis.set_ylim(0.0, 6.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
    axis.set_title(r"MD, $T=0.95$", loc="left", fontsize=11, pad=1.0)


def main():
    arguments = parse_arguments()
    fit_directory = arguments.fit_directory.expanduser().resolve()
    snapshot = arguments.snapshot.expanduser().resolve()
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)

    profiles, summary = load_profiles(fit_directory)
    write_curve_table(output_directory / "panel_d_interface_profiles.csv", profiles)
    (output_directory / "panel_d_interface_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    configure_matplotlib()
    figure = plt.figure(figsize=(3.5, 3.4))
    grid = figure.add_gridspec(
        2,
        1,
        left=0.19,
        right=0.96,
        bottom=0.22,
        top=0.94,
        hspace=0.035,
        height_ratios=(0.72, 1.65),
    )
    snapshot_axis = figure.add_subplot(grid[0])
    axis = figure.add_subplot(grid[1])
    plot_snapshot(snapshot_axis, snapshot)
    temperature_handles = plot_panel(axis, profiles)
    figure.legend(
        handles=temperature_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=1.25,
        handletextpad=0.3,
        columnspacing=0.6,
    )
    figure.text(0.01, 0.985, "d", ha="left", va="top", fontsize=11, weight="bold")
    png = output_directory / "panel_d_interface_profiles.png"
    pdf = output_directory / "panel_d_interface_profiles.pdf"
    figure.savefig(png, dpi=500)
    figure.savefig(pdf)
    plt.close(figure)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
