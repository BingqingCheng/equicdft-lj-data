#!/usr/bin/env python3
"""Plot the shifted-LJ liquid-vapor coexistence curve."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from bulk_response_style import CDFT_COLOR, MD_COLOR


CONTRAST_THRESHOLD = 0.05
MD_CRITICAL_TEMPERATURE = 1.0779
MD_CRITICAL_DENSITY = 0.3190


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


def load_cdft_curve(fit_directory):
    from phase_fit import fit_critical_model

    path = fit_directory / "liquid_vapor_phase_diagram_slab_seed.csv"
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    temperature = np.asarray(rows["temperature"], dtype=float)
    vapor = np.asarray(rows["rho_vapor"], dtype=float)
    liquid = np.asarray(rows["rho_liquid"], dtype=float)
    contrast = np.asarray(rows["density_contrast"], dtype=float)
    converged = np.asarray(rows["converged"], dtype=bool)
    selected = converged & (contrast >= CONTRAST_THRESHOLD)
    order = np.argsort(temperature[selected])

    critical = fit_critical_model(
        temperature,
        vapor,
        liquid,
        converged,
        CONTRAST_THRESHOLD,
    )
    if critical is None:
        raise RuntimeError("Could not fit the cDFT critical point")

    extrapolation_temperature = np.linspace(
        critical["maximum_fitted_temperature"],
        critical["critical_temperature"],
        80,
    )
    gap_squared = (
        critical["gap_slope"] * extrapolation_temperature
        + critical["gap_intercept"]
    )
    gap = np.sqrt(np.maximum(gap_squared, 0.0))
    diameter = (
        critical["diameter_slope"] * extrapolation_temperature
        + critical["diameter_intercept"]
    )
    return {
        "temperature": temperature[selected][order],
        "vapor": vapor[selected][order],
        "liquid": liquid[selected][order],
        "extrapolation_temperature": extrapolation_temperature,
        "extrapolation_vapor": diameter - 0.5 * gap,
        "extrapolation_liquid": diameter + 0.5 * gap,
        "critical_temperature": critical["critical_temperature"],
        "critical_density": critical["critical_density"],
    }


def load_md_reference():
    from thol_eos import coexistence_densities

    # Use the shifted-LJ EOS for the resolved subcritical binodal and close the
    # reference curve at the direct-MD critical point reported by Vrabec et al.
    temperature = np.linspace(0.65, 1.05, 85)
    coexistence = np.asarray(
        [coexistence_densities(value) for value in temperature], dtype=float
    )
    temperature = np.append(temperature, MD_CRITICAL_TEMPERATURE)
    vapor = np.append(coexistence[:, 0], MD_CRITICAL_DENSITY)
    liquid = np.append(coexistence[:, 1], MD_CRITICAL_DENSITY)
    return {
        "temperature": temperature,
        "vapor": vapor,
        "liquid": liquid,
        "critical_temperature": MD_CRITICAL_TEMPERATURE,
        "critical_density": MD_CRITICAL_DENSITY,
    }


def write_curve_table(path, cdft, md):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["source", "segment", "temperature", "rho_vapor", "rho_liquid"]
        )
        for temperature, vapor, liquid in zip(
            cdft["temperature"], cdft["vapor"], cdft["liquid"]
        ):
            writer.writerow(["cDFT", "solved", temperature, vapor, liquid])
        for temperature, vapor, liquid in zip(
            cdft["extrapolation_temperature"],
            cdft["extrapolation_vapor"],
            cdft["extrapolation_liquid"],
        ):
            writer.writerow(["cDFT", "extrapolated", temperature, vapor, liquid])
        for temperature, vapor, liquid in zip(
            md["temperature"], md["vapor"], md["liquid"]
        ):
            writer.writerow(["MD", "reference", temperature, vapor, liquid])


def plot_panel(axis, cdft, md):
    """Draw the liquid-vapor coexistence subpanel."""

    axis.plot(
        md["vapor"][:-1],
        md["temperature"][:-1],
        color=MD_COLOR,
        lw=1.15,
    )
    axis.plot(
        md["liquid"][:-1],
        md["temperature"][:-1],
        color=MD_COLOR,
        lw=1.15,
    )
    axis.scatter(
        cdft["vapor"],
        cdft["temperature"],
        color=CDFT_COLOR,
        marker="o",
        s=14,
        zorder=3,
    )
    axis.scatter(
        cdft["liquid"],
        cdft["temperature"],
        color=CDFT_COLOR,
        marker="o",
        s=14,
        zorder=3,
    )
    axis.plot(
        cdft["extrapolation_vapor"],
        cdft["extrapolation_temperature"],
        color=CDFT_COLOR,
        lw=1.25,
        ls="--",
    )
    axis.plot(
        cdft["extrapolation_liquid"],
        cdft["extrapolation_temperature"],
        color=CDFT_COLOR,
        lw=1.25,
        ls="--",
    )
    axis.scatter(
        cdft["critical_density"],
        cdft["critical_temperature"],
        color=CDFT_COLOR,
        marker="*",
        s=48,
        zorder=4,
    )
    axis.scatter(
        md["critical_density"],
        md["critical_temperature"],
        facecolor="white",
        edgecolor=MD_COLOR,
        linewidth=0.9,
        marker="*",
        s=48,
        zorder=5,
    )

    handles = [
        mlines.Line2D(
            [],
            [],
            color=CDFT_COLOR,
            lw=0.0,
            marker="o",
            markersize=4.0,
            label="cDFT",
        ),
        mlines.Line2D(
            [],
            [],
            color=CDFT_COLOR,
            lw=1.25,
            ls="--",
            label="mean-field fit",
        ),
        mlines.Line2D(
            [], [], color=MD_COLOR, lw=1.15, label="MD"
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        frameon=False,
        fontsize=11,
        handlelength=1.7,
        handletextpad=0.45,
        labelspacing=0.3,
    )
    axis.annotate(
        "MD critical",
        xy=(md["critical_density"], md["critical_temperature"]),
        xytext=(0.32, 1.035),
        fontsize=11,
        color=MD_COLOR,
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": MD_COLOR,
            "lw": 0.8,
            "shrinkA": 2,
            "shrinkB": 5,
        },
    )
    axis.set_xlim(0.0, 0.86)
    axis.set_ylim(0.64, 1.115)
    axis.set_xlabel(r"$\rho\sigma^3$")
    axis.set_ylabel(r"$T$")


def main():
    arguments = parse_arguments()
    fit_directory = arguments.fit_directory.expanduser().resolve()
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    cdft = load_cdft_curve(fit_directory)
    md = load_md_reference()
    write_curve_table(
        output_directory / "panel_c_coexistence_curves.csv", cdft, md
    )
    summary = {
        "cdft": {
            "critical_temperature": float(cdft["critical_temperature"]),
            "critical_density": float(cdft["critical_density"]),
            "n_solved_temperatures": int(cdft["temperature"].size),
        },
        "md_reference": {
            "critical_temperature": float(md["critical_temperature"]),
            "critical_density": float(md["critical_density"]),
        },
    }
    (output_directory / "panel_c_coexistence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(3.5, 2.56))
    figure.subplots_adjust(left=0.18, right=0.97, bottom=0.23, top=0.975)
    plot_panel(axis, cdft, md)
    figure.text(0.01, 0.985, "c", ha="left", va="top", fontsize=11, weight="bold")
    png = output_directory / "panel_c_coexistence_curve.png"
    pdf = output_directory / "panel_c_coexistence_curve.pdf"
    figure.savefig(png, dpi=500)
    figure.savefig(pdf)
    plt.close(figure)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
