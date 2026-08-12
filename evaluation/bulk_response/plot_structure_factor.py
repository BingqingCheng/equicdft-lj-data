#!/usr/bin/env python3
"""Plot low-k MD and learned-cDFT structure factors for selected bulk states."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch

from bulk_response_style import temperature_color


STATES = ((1.2, 0.2), (1.5, 0.5), (1.8, 0.8))
STATE_COLORS = tuple(temperature_color(state[0]) for state in STATES)


def parse_arguments():
    script = Path(__file__).resolve()
    project = script.parents[2]
    default_fit = project / "data" / "bulk_response"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, default=default_fit)
    parser.add_argument(
        "--correlations-directory",
        type=Path,
        default=project / "data" / "bulk_response" / "correlations",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project / "output" / "bulk_response",
    )
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--k-max", type=float, default=3.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--regenerate", action="store_true")
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


def state_directory(root, temperature, density):
    path = root / "nvt-T{:g}-rho{:g}-R1".format(temperature, density)
    if not (path / "sk.csv").is_file():
        matches = sorted(root.glob("nvt-T{:g}-rho{:g}-R*".format(temperature, density)))
        path = next((candidate for candidate in matches if (candidate / "sk.csv").is_file()), None)
    if path is None or not (path / "sk.csv").is_file():
        raise FileNotFoundError(
            "No MD structure factor for T={:g}, rho={:g}".format(
                temperature, density
            )
        )
    return path


def compute_curves(arguments):
    compute_directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(compute_directory))
    from common import load_model, radial_structure_factor

    device = torch.device(arguments.device)
    _model_path, model = load_model(arguments.fit_directory, device)
    grid_size = np.full(3, arguments.grid_size, dtype=int)
    curves = {}
    diagnostics = {}

    for temperature, density in STATES:
        reference_path = state_directory(
            arguments.correlations_directory, temperature, density
        ) / "sk.csv"
        reference = np.genfromtxt(reference_path, delimiter=",", names=True)
        selected = reference["k"] <= arguments.k_max
        reference_curve = np.column_stack(
            [
                reference["k"][selected],
                reference["S_k"][selected],
                reference["standard_error"][selected],
            ]
        )

        rows, result = radial_structure_factor(
            model,
            temperature,
            density,
            grid_size,
            device,
            k_max=arguments.k_max,
        )
        model_curve = np.asarray([[row["k"], row["S_k"]] for row in rows])
        common = (
            (reference_curve[:, 0] >= model_curve[:, 0].min())
            & (reference_curve[:, 0] <= model_curve[:, 0].max())
        )
        observed = reference_curve[common, 1]
        predicted = np.interp(
            reference_curve[common, 0], model_curve[:, 0], model_curve[:, 1]
        )
        error = predicted - observed
        curves[(temperature, density)] = {
            "reference": reference_curve,
            "model": model_curve,
        }
        diagnostics[(temperature, density)] = {
            "temperature": temperature,
            "density": density,
            "grid_size": arguments.grid_size,
            "k_max": arguments.k_max,
            "n_common": int(common.sum()),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "r2": float(
                1.0
                - np.sum(error**2)
                / np.sum((observed - np.mean(observed)) ** 2)
            ),
            "minimum_oz_denominator_full_grid": float(
                result["minimum_oz_denominator"]
            ),
            "max_imaginary_c2_k": float(result["max_imaginary_c2_k"]),
        }
    return curves, diagnostics


def save_cache(path, curves):
    arrays = {}
    for temperature, density in STATES:
        key = "T{:g}_rho{:g}".format(temperature, density).replace(".", "p")
        arrays[key + "_reference"] = curves[(temperature, density)]["reference"]
        arrays[key + "_model"] = curves[(temperature, density)]["model"]
    np.savez_compressed(path, **arrays)


def load_cache(path):
    cache = np.load(path)
    curves = {}
    for temperature, density in STATES:
        key = "T{:g}_rho{:g}".format(temperature, density).replace(".", "p")
        curves[(temperature, density)] = {
            "reference": cache[key + "_reference"],
            "model": cache[key + "_model"],
        }
    return curves


def write_curve_table(path, curves):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["temperature", "density", "source", "k", "S_k", "standard_error"]
        )
        for temperature, density in STATES:
            reference = curves[(temperature, density)]["reference"]
            model = curves[(temperature, density)]["model"]
            for k, structure_factor, standard_error in reference:
                writer.writerow(
                    [temperature, density, "MD", k, structure_factor, standard_error]
                )
            for k, structure_factor in model:
                writer.writerow(
                    [temperature, density, "cDFT", k, structure_factor, ""]
                )


def plot_panel(axis, curves, k_max):
    """Draw the structure-factor subpanel on an existing Matplotlib axis."""

    for (temperature, density), color in zip(STATES, STATE_COLORS):
        reference = curves[(temperature, density)]["reference"]
        model = curves[(temperature, density)]["model"]
        axis.fill_between(
            reference[:, 0],
            reference[:, 1] - reference[:, 2],
            reference[:, 1] + reference[:, 2],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        axis.plot(
            reference[:, 0],
            reference[:, 1],
            color=color,
            lw=1.25,
        )
        axis.plot(
            model[:, 0],
            model[:, 1],
            color=color,
            lw=1.0,
            ls="--",
            marker="o",
            ms=2.8,
            markerfacecolor="white",
            markeredgewidth=0.7,
        )

    state_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            lw=1.4,
            label=r"$T={:g},\;\rho={:g}$".format(temperature, density),
        )
        for (temperature, density), color in zip(STATES, STATE_COLORS)
    ]
    method_handles = [
        mlines.Line2D([], [], color="0.2", lw=1.25, label="MD"),
        mlines.Line2D(
            [],
            [],
            color="0.2",
            lw=1.0,
            ls="--",
            marker="o",
            ms=3.0,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label="cDFT",
        ),
    ]
    state_legend = axis.legend(
        handles=state_handles,
        loc="upper right",
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.28,
    )
    axis.add_artist(state_legend)
    axis.legend(
        handles=method_handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.43),
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.28,
    )
    axis.set_xlim(0.0, k_max)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel(r"$k\sigma$")
    axis.set_ylabel(r"$S(k)$")


def main():
    arguments = parse_arguments()
    arguments.fit_directory = arguments.fit_directory.expanduser().resolve()
    arguments.correlations_directory = (
        arguments.correlations_directory.expanduser().resolve()
    )
    arguments.output_directory = arguments.output_directory.expanduser().resolve()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    source = repository / "source_data" / "fig02_bulk_response"
    cache = (
        arguments.output_directory / "panel_a_structure_factor_points.npz"
        if arguments.regenerate
        else source / "panel_a_structure_factor_points.npz"
    )
    summary_path = arguments.output_directory / "panel_a_structure_factor_summary.json"

    if arguments.regenerate or not cache.is_file():
        curves, diagnostics = compute_curves(arguments)
        save_cache(cache, curves)
        summary_path.write_text(
            json.dumps(list(diagnostics.values()), indent=2) + "\n"
        )
    else:
        curves = load_cache(cache)

    write_curve_table(
        arguments.output_directory / "panel_a_structure_factor_curves.csv", curves
    )
    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.5, 2.56))
    figure.subplots_adjust(left=0.19, right=0.94, bottom=0.24, top=0.975)
    plot_panel(axis, curves, arguments.k_max)
    figure.text(0.01, 0.985, "a", ha="left", va="top", fontsize=11, weight="bold")
    png = arguments.output_directory / "panel_a_structure_factor.png"
    pdf = arguments.output_directory / "panel_a_structure_factor.pdf"
    figure.savefig(png, dpi=500)
    figure.savefig(pdf)
    plt.close(figure)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
