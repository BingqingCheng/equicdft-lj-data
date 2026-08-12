#!/usr/bin/env python3
"""Combine the four parity benchmarks into one two-column 2 x 2 figure."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HELD_OUT_TEMPERATURES = (0.7, 1.1, 1.5)
GCMC_TEMPERATURES = (1.0, 1.2, 1.4)
TEMPERATURE_COLORS = ("#C84A3A", "#238A8D", "#3D5A98")


def parse_arguments():
    script = Path(__file__).resolve()
    repository = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=repository / "source_data" / "fig05_combined_benchmarks",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=repository / "output"
    )
    return parser.parse_args()


def rmse_r2(target, prediction):
    error = prediction - target
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float("nan") if denominator <= 0.0 else 1.0 - float(
        np.sum(error**2)
    ) / denominator
    return rmse, r2


def square_limits(point_sets, lower_quantile=0.0, upper_quantile=1.0):
    values = np.concatenate(
        [array for target, prediction in point_sets.values() for array in (target, prediction)]
    )
    lower, upper = np.quantile(values, [lower_quantile, upper_quantile])
    span = upper - lower
    padding = 0.04 * span if span > 0.0 else 0.1
    return float(lower - padding), float(upper + padding)


def load_figure1(data_directory):
    forward = {}
    solve = {}
    for temperature in HELD_OUT_TEMPERATURES:
        key = str(temperature).replace(".", "p")
        tag = f"_T{key}"
        cache = np.load(
            data_directory / ("complete_small_nvt" + tag + "_points.npz")
        )
        metrics = json.loads(
            (
                data_directory
                / ("complete_small_nvt" + tag + "_metrics.json")
            ).read_text()
        )
        row = metrics[f"T={temperature:.1f}"]
        forward[temperature] = {
            "display": (
                cache[f"T{key}_forward_target"],
                cache[f"T{key}_forward_prediction"],
            ),
            "rmse": float(row["forward"]["rmse"]),
            "r2": float(row["forward"]["r2"]),
        }
        solve[temperature] = {
            "display": (
                cache[f"T{key}_solve_target"],
                cache[f"T{key}_solve_prediction"],
            ),
            "rmse": float(row["solve"]["rmse"]),
            "r2": float(row["solve"]["r2"]),
        }
    return forward, solve


def load_large_nvt(data_directory):
    output = {}
    for temperature in HELD_OUT_TEMPERATURES:
        key = str(temperature).replace(".", "p")
        tag = f"_T{key}"
        points_path = data_directory / (
            "complete_large_nvt" + tag + "_points.npz"
        )
        metrics_path = data_directory / (
            "complete_large_nvt" + tag + "_metrics.json"
        )
        if not points_path.is_file():
            points_path = data_directory / "complete_large_nvt_points.npz"
            metrics_path = data_directory / "complete_large_nvt_metrics.json"
        cache = np.load(points_path)
        metrics = json.loads(
            metrics_path.read_text()
        )
        row = metrics[f"T={temperature:.1f}"]
        output[temperature] = {
            "display": (
                cache[f"T{key}_target"],
                cache[f"T{key}_prediction"],
            ),
            "rmse": float(row["rmse"]),
            "r2": float(row["r2"]),
        }
    return output


def load_gcmc(data_directory):
    rows = list(csv.DictReader((data_directory / "gcmc_mu_predictions.csv").open()))
    output = {}
    for temperature in GCMC_TEMPERATURES:
        selected = [
            row
            for row in rows
            if np.isclose(float(row["temperature"]), temperature)
        ]
        reference = np.asarray(
            [float(row["beta_mu_reference"]) for row in selected], dtype=float
        )
        prediction = np.asarray(
            [float(row["beta_mu_prediction_raw"]) for row in selected], dtype=float
        )
        reference -= np.mean(reference)
        prediction -= np.mean(prediction)
        rmse, r2 = rmse_r2(reference, prediction)
        output[temperature] = {
            "display": (reference, prediction),
            "complete": (reference, prediction),
            "rmse": rmse,
            "r2": r2,
        }
    return output


def add_panel_label(axis, label):
    axis.text(
        -0.17,
        1.01,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )


def add_benchmark_label(axis, label):
    axis.text(
        0.97,
        0.035,
        label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "square,pad=0.20",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        zorder=10,
    )


def draw_panel(
    axis,
    data,
    temperatures,
    xlabel,
    ylabel,
    limits,
    benchmark_label,
    marker_size,
    alpha,
    r2_decimals,
):
    for temperature, color in zip(temperatures, TEMPERATURE_COLORS):
        target, prediction = data[temperature]["display"]
        label = (
            r"$T={:.1f}$" "\n" + r"RMSE={:.2f}, $R^2$={:." + str(r2_decimals) + "f}"
        ).format(
            temperature,
            data[temperature]["rmse"],
            data[temperature]["r2"],
        )
        axis.scatter(
            target,
            prediction,
            s=marker_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            color=color,
            label=label,
        )
    axis.plot(limits, limits, color="#252525", lw=1.0, ls="--", zorder=5)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(
        frameon=True,
        fontsize=10,
        loc="upper left",
        handletextpad=0.20,
        borderaxespad=0.25,
        borderpad=0.25,
        labelspacing=0.42,
        markerscale=2.0,
        facecolor="white",
        framealpha=0.72,
        edgecolor="none",
    )
    add_benchmark_label(axis, benchmark_label)


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


def main():
    arguments = parse_arguments()
    data_directory = arguments.data_directory.expanduser().resolve()
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    forward, solve = load_figure1(data_directory)
    large_nvt = load_large_nvt(data_directory)
    gcmc = load_gcmc(data_directory)

    configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.09,
        top=0.945,
        wspace=0.29,
        hspace=0.30,
    )

    forward_points = {
        temperature: data["display"] for temperature, data in forward.items()
    }
    draw_panel(
        axes[0, 0],
        forward,
        HELD_OUT_TEMPERATURES,
        r"Reference $\beta V^{\mathrm{ext}}$",
        r"Predicted $\beta V^{\mathrm{ext}}$",
        square_limits(forward_points, lower_quantile=0.002, upper_quantile=0.998),
        r"NVT: $\rho\rightarrow V^{\mathrm{ext}}$",
        marker_size=3.5,
        alpha=0.13,
        r2_decimals=2,
    )
    add_panel_label(axes[0, 0], "a")

    solve_points = {
        temperature: data["display"] for temperature, data in solve.items()
    }
    solve_limits = square_limits(solve_points, upper_quantile=0.997)
    solve_limits = (max(0.0, solve_limits[0]), solve_limits[1])
    draw_panel(
        axes[0, 1],
        solve,
        HELD_OUT_TEMPERATURES,
        r"Reference $\rho\sigma^3$",
        r"Solved $\rho\sigma^3$",
        solve_limits,
        r"NVT: $V^{\mathrm{ext}}\rightarrow\rho$",
        marker_size=3.5,
        alpha=0.13,
        r2_decimals=2,
    )
    add_panel_label(axes[0, 1], "b")

    large_display = {
        temperature: data["display"] for temperature, data in large_nvt.items()
    }
    large_limits = square_limits(large_display, upper_quantile=0.9995)
    large_limits = (max(0.0, large_limits[0]), large_limits[1])
    draw_panel(
        axes[1, 0],
        large_nvt,
        HELD_OUT_TEMPERATURES,
        r"Reference $\rho\sigma^3$",
        r"Solved $\rho\sigma^3$",
        large_limits,
        r"NVT: small$\rightarrow$large",
        marker_size=3.5,
        alpha=0.13,
        r2_decimals=3,
    )
    add_panel_label(axes[1, 0], "c")

    gcmc_complete = {
        temperature: data["complete"] for temperature, data in gcmc.items()
    }
    draw_panel(
        axes[1, 1],
        gcmc,
        GCMC_TEMPERATURES,
        r"Reference $\Delta(\beta\mu)$",
        r"Predicted $\Delta(\beta\mu)$",
        square_limits(gcmc_complete),
        r"NVT$\rightarrow$GCMC",
        marker_size=16.0,
        alpha=0.65,
        r2_decimals=3,
    )
    add_panel_label(axes[1, 1], "d")

    stem = output_directory / "combined_benchmark_2x2"
    figure.savefig(stem.with_suffix(".png"), dpi=500)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    print("wrote", stem.with_suffix(".png"))
    print("wrote", stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
