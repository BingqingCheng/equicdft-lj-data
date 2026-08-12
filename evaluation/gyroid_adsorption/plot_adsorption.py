#!/usr/bin/env python3
"""Plot the two-panel gyroid adsorption figure for one cDFT model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_DATA = REPOSITORY / "source_data" / "fig04_gyroid_adsorption"
DEFAULT_GCMC = DEFAULT_DATA / "gcmc_mu_density.csv"


def read_table(path: Path) -> np.ndarray:
    """Read a named-column CSV table."""

    return np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")


def configure_style(font_size: float) -> None:
    """Use compact journal-style typography with a single requested size."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size - 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
        }
    )


def plot_figure(
    results_directory: Path,
    gcmc_path: Path,
    output_path: Path,
    font_size: float,
) -> None:
    """Create chemical-potential and equal-mu adsorption panels."""

    cdft = read_table(results_directory / "cdft_mu_density.csv")
    adsorption = read_table(results_directory / "adsorption_isotherm.csv")
    gcmc = read_table(gcmc_path)

    configure_style(font_size)
    blue = "#0072B2"
    orange = "#D55E00"
    neutral = "#222222"

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.50, 2.80),
        gridspec_kw={"width_ratios": (1.50, 0.80)},
        constrained_layout=True,
    )

    # (a) The homogeneous and confined equations of state in the same mu gauge.
    axis = axes[0]
    axis.plot(
        cdft["rho_bulk_gcmc"],
        cdft["mu_cdft_bulk"],
        color=blue,
        label="Bulk cDFT",
        zorder=2,
    )
    axis.plot(
        cdft["rho_gyroid_box_gcmc"],
        cdft["mu_cdft_gyroid"],
        color=orange,
        label="Gyroid cDFT",
        zorder=2,
    )
    axis.errorbar(
        gcmc["rho_bulk"],
        gcmc["mu"],
        xerr=gcmc["rho_bulk_sem"],
        linestyle="none",
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        color=blue,
        capsize=1.8,
        label="Bulk GCMC",
        zorder=3,
    )
    axis.errorbar(
        gcmc["rho_gyroid_box"],
        gcmc["mu"],
        xerr=gcmc["rho_gyroid_box_sem"],
        linestyle="none",
        marker="s",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        color=orange,
        capsize=1.8,
        label="Gyroid GCMC",
        zorder=3,
    )
    axis.set_xlabel(r"$\rho\sigma^3$")
    axis.set_ylabel(r"$\mu/\epsilon$")
    axis.set_xlim(0.0, 0.85)
    axis.set_ylim(-4.4, 1.35)
    handles, labels = axis.get_legend_handles_labels()
    # Two-column, table-like legend:
    # Bulk GCMC | Bulk cDFT
    # Gyroid GCMC | Gyroid cDFT
    order = [2, 3, 0, 1]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="lower right",
        ncols=2,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
        handletextpad=0.5,
    )
    axis.text(
        0.04,
        0.96,
        r"$T=1.2$",
        transform=axis.transAxes,
        va="top",
    )
    axis.text(
        -0.065,
        1.02,
        "d",
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontweight="bold",
        clip_on=False,
    )

    # (b) Adsorption follows by matching bulk and pore states at equal mu.
    axis = axes[1]
    axis.plot(
        adsorption["rho_bulk"],
        adsorption["rho_gyroid_box"],
        color=blue,
        label="cDFT",
        zorder=2,
    )
    axis.errorbar(
        gcmc["rho_bulk"],
        gcmc["rho_gyroid_box"],
        xerr=gcmc["rho_bulk_sem"],
        yerr=gcmc["rho_gyroid_box_sem"],
        linestyle="none",
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor=neutral,
        markeredgewidth=1.2,
        ecolor=neutral,
        capsize=1.8,
        label="GCMC",
        zorder=3,
    )
    axis.set_xlabel(r"$\rho_{\mathrm{b}}\sigma^3$")
    axis.set_ylabel(r"$\rho_{\mathrm{gyr}}\sigma^3$")
    axis.set_xlim(0.0, 0.85)
    axis.set_ylim(0.0, 0.40)
    axis.set_xticks(np.arange(0.0, 0.81, 0.20))
    axis.set_yticks(np.arange(0.0, 0.41, 0.10))
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.04, 0.90),
        borderaxespad=0.0,
        frameon=False,
        handlelength=2.1,
    )
    axis.text(
        -0.20,
        1.02,
        "e",
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontweight="bold",
        clip_on=False,
    )

    for axis in axes:
        axis.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
            width=0.8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=400)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_directory",
        type=Path,
        nargs="?",
        default=DEFAULT_DATA,
        help="directory containing the gyroid source-data CSVs",
    )
    parser.add_argument("--gcmc", type=Path, default=DEFAULT_GCMC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font-size", type=float, default=11.0)
    arguments = parser.parse_args()

    results_directory = arguments.results_directory.expanduser().resolve()
    output = arguments.output
    if output is None:
        output = REPOSITORY / "output" / "gyroid_adsorption_two_panel_T1.2.png"
    plot_figure(
        results_directory=results_directory,
        gcmc_path=arguments.gcmc.expanduser().resolve(),
        output_path=output.expanduser().resolve(),
        font_size=arguments.font_size,
    )
    print("wrote {} and {}".format(output, output.with_suffix(".pdf")))


if __name__ == "__main__":
    main()
