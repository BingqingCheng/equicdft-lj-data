#!/usr/bin/env python3
"""Plot the shifted-LJ equation of state from cDFT and the MD reference EOS."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from bulk_response_style import temperature_color


TEMPERATURES = (0.8, 1.0, 1.2, 1.5, 1.8)
TEMPERATURE_COLORS = tuple(temperature_color(value) for value in TEMPERATURES)


def parse_arguments():
    script = Path(__file__).resolve()
    project = script.parents[2]
    default_fit = project / "data" / "bulk_response"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, default=default_fit)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project / "output" / "bulk_response",
    )
    parser.add_argument("--rho-max", type=float, default=0.9)
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


def pressure_path(directory, temperature):
    candidates = (
        directory / "pressure_from_S0_T{:g}.csv".format(temperature),
        directory / "pressure_from_S0_T{:.1f}.csv".format(temperature),
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    return path


def load_curves(arguments):
    from thol_eos import CRITICAL_TEMPERATURE, coexistence_densities, evaluate

    curves = {}
    summary = []
    for temperature in TEMPERATURES:
        model = np.genfromtxt(
            pressure_path(arguments.fit_directory, temperature),
            delimiter=",",
            names=True,
        )
        model = model[model["density"] <= arguments.rho_max]
        thol_pressure = np.asarray(
            evaluate(temperature, model["density"])["pressure"]
        ).copy()
        coexistence = None
        if temperature < CRITICAL_TEMPERATURE:
            vapor, liquid = coexistence_densities(temperature)
            coexistence_pressure = float(evaluate(temperature, vapor)["pressure"])
            in_two_phase_region = (
                (model["density"] > vapor) & (model["density"] < liquid)
            )
            thol_pressure[in_two_phase_region] = coexistence_pressure
            coexistence = {
                "vapor_density": float(vapor),
                "liquid_density": float(liquid),
                "coexistence_pressure": coexistence_pressure,
            }

        error = model["pressure"] - thol_pressure
        rmse = float(np.sqrt(np.mean(error**2)))
        centered = thol_pressure - np.mean(thol_pressure)
        denominator = float(np.sum(centered**2))
        r2 = (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0.0
            else None
        )
        summary.append(
            {
                "temperature": temperature,
                "n_density_points": int(model.size),
                "rmse_cdft_vs_md_reference_eos": rmse,
                "r2_cdft_vs_md_reference_eos": r2,
                "coexistence": coexistence,
            }
        )
        curves[temperature] = {
            "model_density": model["density"],
            "model_pressure": model["pressure"],
            "thol_pressure": thol_pressure,
        }
    return curves, summary


def write_curve_table(path, curves):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["temperature", "source", "density", "pressure", "standard_error"]
        )
        for temperature in TEMPERATURES:
            curve = curves[temperature]
            for density, pressure in zip(
                curve["model_density"], curve["model_pressure"]
            ):
                writer.writerow([temperature, "cDFT", density, pressure, ""])
            for density, pressure in zip(
                curve["model_density"], curve["thol_pressure"]
            ):
                writer.writerow([temperature, "MD", density, pressure, ""])


def plot_panel(axis, curves, rho_max):
    """Draw the shifted-LJ equation-of-state subpanel."""

    for temperature, color in zip(TEMPERATURES, TEMPERATURE_COLORS):
        curve = curves[temperature]
        axis.plot(
            curve["model_density"],
            curve["model_pressure"],
            color=color,
            lw=1.0,
            ls="--",
        )
        axis.plot(
            curve["model_density"],
            curve["thol_pressure"],
            color=color,
            lw=1.35,
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
            [], [], color=color, lw=1.6, label=r"$T={:.1f}$".format(temperature)
        )
        for temperature, color in zip(TEMPERATURES, TEMPERATURE_COLORS)
    ]
    axis.legend(
        handles=temperature_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.74),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    axis.axhline(0.0, color="0.55", lw=0.7, zorder=0)
    axis.set_xlim(0.0, rho_max)
    axis.set_xlabel(r"$\rho\sigma^3$")
    axis.set_ylabel(r"$P\sigma^3/\epsilon$")


def main():
    arguments = parse_arguments()
    arguments.fit_directory = arguments.fit_directory.expanduser().resolve()
    arguments.output_directory = arguments.output_directory.expanduser().resolve()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    curves, summary = load_curves(arguments)
    write_curve_table(
        arguments.output_directory / "panel_b_equation_of_state_curves.csv", curves
    )
    (
        arguments.output_directory / "panel_b_equation_of_state_summary.json"
    ).write_text(json.dumps(summary, indent=2) + "\n")

    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.5, 3.2))
    figure.subplots_adjust(left=0.19, right=0.94, bottom=0.19, top=0.98)
    plot_panel(axis, curves, arguments.rho_max)
    figure.text(0.01, 0.985, "b", ha="left", va="top", fontsize=11, weight="bold")
    png = arguments.output_directory / "panel_b_equation_of_state.png"
    pdf = arguments.output_directory / "panel_b_equation_of_state.pdf"
    figure.savefig(png, dpi=500)
    figure.savefig(pdf)
    plt.close(figure)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
