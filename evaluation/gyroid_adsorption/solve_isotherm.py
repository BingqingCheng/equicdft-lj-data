#!/usr/bin/env python3
"""Evaluate a cDFT model on the soft-gyroid adsorption isotherm.

One command performs the complete model-side workflow:

1. recover every gyroid GCMC density field with a fixed-N cDFT solve;
2. evaluate the model chemical potential of the homogeneous fluid at the
   matched GCMC bulk densities;
3. determine one additive chemical-potential gauge from the homogeneous
   series and apply it unchanged to both homogeneous and gyroid results;
4. write solved fields, a compact table, metrics, and comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

# Work on batch nodes and restricted workstations without a writable home
# cache. Existing user-provided cache locations still take precedence.
_CACHE_DIRECTORY = Path(tempfile.gettempdir()) / "equicdft-plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIRECTORY / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIRECTORY))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read, write


APPLICATION_DIRECTORY = Path(__file__).resolve().parent
WORKSPACE = APPLICATION_DIRECTORY.parents[1]
DEFAULT_GYROID_CAMPAIGN = (
    WORKSPACE / "data" / "gyroid_adsorption" / "gyroid_gcmc"
)
DEFAULT_GCMC_TABLE = WORKSPACE / "source_data" / "fig04_gyroid_adsorption" / "gcmc_mu_density.csv"


def scalar(value):
    """Convert a scalar tensor or number to a Python float."""

    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def to_device(data, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def regression_metrics(reference, prediction):
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - reference
    mse = float(np.mean(error**2))
    variance = float(np.mean((reference - np.mean(reference)) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "r2": float("nan") if variance == 0.0 else 1.0 - mse / variance,
        "pearson_r": float(np.corrcoef(reference, prediction)[0, 1]),
    }


def read_gcmc_table(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("empty GCMC table: {}".format(path))
    return sorted(rows, key=lambda row: float(row["mu"]))


def gyroid_inputs(campaign):
    paths = campaign.glob(
        "mu-*/run-001/density_external_grid_0.5sigma.extxyz"
    )
    records = []
    for path in paths:
        chemical_potential = float(read(path, index=0).info["mu"])
        records.append((chemical_potential, path))
    if not records:
        raise FileNotFoundError("no gyroid grid fields found in {}".format(campaign))
    return dict(records)


def mu_label(chemical_potential):
    return ("{:+.1f}".format(chemical_potential)).replace(
        "+", "p"
    ).replace("-", "m").replace(".", "p")


def solve_gyroid_state(
    input_path,
    output_path,
    model,
    solver,
    device,
    maximum_density,
    max_iter,
    tolerance_residual,
    tolerance_rms_residual,
    tolerance_change,
):
    """Recover one GCMC density field using only V_ext, T, and its mean N."""

    from equicdft import GridData

    atoms = read(input_path, index=0)
    data = to_device(
        GridData.from_xyz(input_path, grid_info=model.grid_info)[0],
        device,
    )
    reference_rho = data["rho"]
    solve_data = dict(data)
    solve_data.pop("rho")
    solve_data.pop("mu", None)
    solve_data.pop("beta_mu", None)

    cell_volume = torch.prod(solve_data["grid_spacing"])
    particle_numbers = cell_volume * torch.sum(reference_rho, dim=0)
    solved = solver.solve(
        solve_data,
        particle_numbers=particle_numbers,
        method="euler",
        beta_multiplier=0.0,
        max_iter=max_iter,
        tolerance_residual=tolerance_residual,
        tolerance_rms_residual=tolerance_rms_residual,
        tolerance_change=tolerance_change,
        residual_density_threshold=float(model.rho_min),
        maximum_density=maximum_density,
    )

    reference = reference_rho[:, 0].detach().cpu().numpy()
    prediction = solved["rho"][:, 0].detach().cpu().numpy()
    density_metrics = regression_metrics(reference, prediction)
    beta = scalar(data["beta"])

    # Restore the original EXTXYZ row ordering before writing the solved field.
    grid_size = np.asarray(atoms.info["grid_size"], dtype=int)
    grid_positions = np.rint(atoms.get_positions()).astype(int)
    if str(atoms.info.get("grid_indexing", "zero_based")).lower().startswith(
        "one"
    ):
        grid_positions -= 1
    flat_positions = np.ravel_multi_index(
        grid_positions.T,
        tuple(grid_size),
        order="C",
    )
    reference_atoms_order = reference[flat_positions]
    prediction_atoms_order = prediction[flat_positions]
    atoms.set_array("reference_density", reference_atoms_order)
    atoms.set_array("density", prediction_atoms_order)
    atoms.set_array(
        "density_error",
        prediction_atoms_order - reference_atoms_order,
    )

    physical_spacing = np.asarray(atoms.info["grid_spacing"], dtype=float)
    physical_volume = float(np.prod(grid_size * physical_spacing))
    physical_cell_volume = float(np.prod(physical_spacing))
    rho_gyroid_box = float(
        np.sum(prediction_atoms_order) * physical_cell_volume / physical_volume
    )
    mu_gyroid_raw = scalar(solved["average_chemical_potential"][0]) / beta

    atoms.info.update(
        {
            "model": model.__class__.__name__,
            "solve_method": "adaptive_euler_fixed_N",
            "solve_converged": bool(solved["converged"]),
            "solve_iterations": int(solved["n_iter"]),
            "particle_number_fixed": scalar(particle_numbers[0]),
            "mu_cdft_raw": mu_gyroid_raw,
        }
    )
    write(output_path, atoms, format="extxyz")

    return {
        "mu_gyroid_raw": mu_gyroid_raw,
        "rho_gyroid_box_cdft": rho_gyroid_box,
        "particle_number": scalar(particle_numbers[0]),
        "converged": bool(solved["converged"]),
        "iterations": int(solved["n_iter"]),
        "max_residual": scalar(solved["max_euler_lagrange_residual"]),
        "rms_residual": scalar(solved["rms_euler_lagrange_residual"]),
        "density_mae": density_metrics["mae"],
        "density_rmse": density_metrics["rmse"],
        "density_r2": density_metrics["r2"],
        "density_pearson_r": density_metrics["pearson_r"],
        "solved_field": str(output_path.resolve()),
        "grid_spacing": physical_spacing.tolist(),
    }


def add_homogeneous_mu(rows, model, solver, device, bulk_box_length):
    """Evaluate mu for uniform fluids at the GCMC bulk densities."""

    from equicdft import GridData

    spacing = np.asarray(rows[0]["grid_spacing"], dtype=float)
    grid_size = np.rint(bulk_box_length / spacing).astype(int)
    if not np.allclose(grid_size * spacing, bulk_box_length):
        raise ValueError("bulk box length must be commensurate with grid spacing")

    base = GridData.from_dict(
        {
            "grid_size": grid_size.tolist(),
            "grid_spacing": spacing.tolist(),
            "temperature": rows[0]["temperature"],
            "n_types": 1,
        },
        grid_info=model.grid_info,
    )
    base = to_device(base, device)
    n_grid = int(base["grid_positions"].shape[0])
    dtype = model.grid_spacing.dtype

    for row in rows:
        data = dict(base)
        data["rho"] = torch.full(
            (n_grid, 1),
            row["rho_bulk_gcmc"],
            dtype=dtype,
            device=device,
        )
        data["V_ext"] = torch.zeros_like(data["rho"])
        evaluated = solver.evaluate(data, compute_c1=True)
        beta = scalar(data["beta"])
        row["mu_bulk_raw"] = (
            scalar(evaluated["average_chemical_potential"][0]) / beta
        )


def align_common_mu_gauge(rows):
    """Calibrate one offset on bulk states and apply it to both systems."""

    reference = np.asarray([row["mu_gcmc"] for row in rows])
    bulk_raw = np.asarray([row["mu_bulk_raw"] for row in rows])
    gyroid_raw = np.asarray([row["mu_gyroid_raw"] for row in rows])
    offset = float(np.mean(reference - bulk_raw))
    for row in rows:
        row["mu_gauge_offset"] = offset
        row["mu_cdft_bulk"] = row["mu_bulk_raw"] + offset
        row["mu_cdft_gyroid"] = row["mu_gyroid_raw"] + offset

    return {
        "mu_gauge_source": "homogeneous GCMC density series",
        "mu_gauge_offset": offset,
        "bulk_mu": regression_metrics(reference, bulk_raw + offset),
        "gyroid_mu": regression_metrics(reference, gyroid_raw + offset),
        "gyroid_density_mean_rmse": float(
            np.mean([row["density_rmse"] for row in rows])
        ),
        "gyroid_density_max_rmse": float(
            np.max([row["density_rmse"] for row in rows])
        ),
        "converged_states": int(sum(row["converged"] for row in rows)),
        "states": len(rows),
    }


def write_table(path, rows):
    columns = [
        "mu_gcmc",
        "temperature",
        "mu_gauge_offset",
        "rho_bulk_gcmc",
        "mu_cdft_bulk",
        "mu_bulk_raw",
        "rho_gyroid_box_gcmc",
        "rho_gyroid_box_cdft",
        "mu_cdft_gyroid",
        "mu_gyroid_raw",
        "particle_number",
        "converged",
        "iterations",
        "max_residual",
        "rms_residual",
        "density_mae",
        "density_rmse",
        "density_r2",
        "density_pearson_r",
        "solved_field",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in rows])


def plot_cdft(path, rows):
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    ax.plot(
        [row["mu_cdft_bulk"] for row in rows],
        [row["rho_bulk_gcmc"] for row in rows],
        "o-",
        color="#1f4e79",
        markersize=5.5,
        linewidth=1.7,
        label=r"homogeneous bulk, $L=8\sigma$",
    )
    ax.plot(
        [row["mu_cdft_gyroid"] for row in rows],
        [row["rho_gyroid_box_cdft"] for row in rows],
        "^-",
        color="#c45a18",
        markersize=6.0,
        linewidth=1.7,
        label=r"gyroid box average, $L=16\sigma$",
    )
    format_axes(ax, rows, "cDFT chemical potential")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_comparison(path, rows):
    fig, ax = plt.subplots(figsize=(5.7, 4.3))
    mu_gcmc = [row["mu_gcmc"] for row in rows]
    ax.errorbar(
        mu_gcmc,
        [row["rho_bulk_gcmc"] for row in rows],
        yerr=[row["rho_bulk_sem"] for row in rows],
        color="#1f4e79",
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markersize=5.5,
        linewidth=1.4,
        capsize=2.0,
        label="bulk: GCMC",
    )
    ax.plot(
        [row["mu_cdft_bulk"] for row in rows],
        [row["rho_bulk_gcmc"] for row in rows],
        "o-",
        color="#1f4e79",
        markersize=4.5,
        linewidth=1.8,
        label="bulk: cDFT",
    )
    ax.errorbar(
        mu_gcmc,
        [row["rho_gyroid_box_gcmc"] for row in rows],
        yerr=[row["rho_gyroid_box_sem"] for row in rows],
        color="#c45a18",
        linestyle="--",
        marker="^",
        markerfacecolor="white",
        markersize=6.0,
        linewidth=1.4,
        capsize=2.0,
        label="gyroid box: GCMC",
    )
    ax.plot(
        [row["mu_cdft_gyroid"] for row in rows],
        [row["rho_gyroid_box_cdft"] for row in rows],
        "^-",
        color="#c45a18",
        markersize=5.0,
        linewidth=1.8,
        label="gyroid box: cDFT",
    )
    format_axes(ax, rows, "chemical potential (common gauge)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def build_adsorption_isotherm(rows, n_points=401):
    """Eliminate mu between the bulk and gyroid equations of state."""

    mu_bulk = np.asarray([row["mu_cdft_bulk"] for row in rows])
    rho_bulk = np.asarray([row["rho_bulk_gcmc"] for row in rows])
    mu_gyroid = np.asarray([row["mu_cdft_gyroid"] for row in rows])
    rho_gyroid = np.asarray([row["rho_gyroid_box_cdft"] for row in rows])

    bulk_order = np.argsort(mu_bulk)
    gyroid_order = np.argsort(mu_gyroid)
    mu_bulk = mu_bulk[bulk_order]
    rho_bulk = rho_bulk[bulk_order]
    mu_gyroid = mu_gyroid[gyroid_order]
    rho_gyroid = rho_gyroid[gyroid_order]
    if np.any(np.diff(mu_bulk) <= 0.0) or np.any(np.diff(mu_gyroid) <= 0.0):
        raise ValueError("chemical potential must be monotonic for inversion")

    mu_min = max(mu_bulk[0], mu_gyroid[0])
    mu_max = min(mu_bulk[-1], mu_gyroid[-1])
    if mu_max <= mu_min:
        raise ValueError("bulk and gyroid chemical-potential ranges do not overlap")
    mu_common = np.linspace(mu_min, mu_max, n_points)
    return {
        "mu_common": mu_common,
        "rho_bulk": np.interp(mu_common, mu_bulk, rho_bulk),
        "rho_gyroid_box": np.interp(
            mu_common,
            mu_gyroid,
            rho_gyroid,
        ),
        "rho_gyroid_at_reference_bulk": np.interp(
            np.asarray([row["mu_cdft_bulk"] for row in rows]),
            mu_gyroid,
            rho_gyroid,
        ),
    }


def write_adsorption_table(path, adsorption):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["mu_common", "rho_bulk", "rho_gyroid_box"])
        writer.writerows(
            zip(
                adsorption["mu_common"],
                adsorption["rho_bulk"],
                adsorption["rho_gyroid_box"],
            )
        )


def plot_adsorption_isotherm(path, rows, adsorption):
    """Plot gyroid loading against bulk density at equal model mu."""

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(
        [row["rho_bulk_gcmc"] for row in rows],
        [row["rho_gyroid_box_gcmc"] for row in rows],
        "o--",
        color="#555555",
        markerfacecolor="white",
        markersize=5.5,
        linewidth=1.4,
        label=r"GCMC: matched $\mu$",
    )
    ax.plot(
        adsorption["rho_bulk"],
        adsorption["rho_gyroid_box"],
        color="#c45a18",
        linewidth=2.0,
        label=r"cDFT: $\mu_{\rm b}=\mu_{\rm gyr}$",
    )
    ax.plot(
        [row["rho_bulk_gcmc"] for row in rows],
        adsorption["rho_gyroid_at_reference_bulk"],
        "^",
        color="#c45a18",
        markersize=5.0,
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"homogeneous density, $\rho_{\rm b}\sigma^3$")
    ax.set_ylabel(r"gyroid whole-box density, $\rho_{\rm gyr}\sigma^3$")
    ax.text(
        0.04,
        0.96,
        r"$T^*={:.3g}$".format(rows[0]["temperature"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.grid(color="0.88", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def format_axes(ax, rows, chemical_potential_label):
    all_mu = np.asarray(
        [row["mu_gcmc"] for row in rows]
        + [row["mu_cdft_bulk"] for row in rows]
        + [row["mu_cdft_gyroid"] for row in rows]
    )
    ax.set_xlim(all_mu.min() - 0.12, all_mu.max() + 0.12)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(chemical_potential_label + r", $\mu/\epsilon$")
    ax.set_ylabel(r"number density, $\rho\sigma^3$")
    ax.text(
        0.03,
        0.96,
        r"$T^*={:.3g}$".format(rows[0]["temperature"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(color="0.88", linewidth=0.7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=WORKSPACE / "model" / "model.pt", help="trained cDFT model.pt")
    parser.add_argument(
        "--gyroid-campaign",
        type=Path,
        default=DEFAULT_GYROID_CAMPAIGN,
    )
    parser.add_argument("--gcmc-table", type=Path, default=DEFAULT_GCMC_TABLE)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bulk-box-length", type=float, default=8.0)
    parser.add_argument("--maximum-density", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tolerance-residual", type=float, default=3.0e-2)
    parser.add_argument("--tolerance-rms-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-5)
    arguments = parser.parse_args()

    from equicdft import GridSolver

    model_path = arguments.model.expanduser().resolve()
    output_directory = arguments.output_directory
    if output_directory is None:
        output_directory = (
            WORKSPACE / "output" / "gyroid_adsorption"
        )
    output_directory = output_directory.expanduser().resolve()
    fields_directory = output_directory / "solved-fields"
    fields_directory.mkdir(parents=True, exist_ok=True)

    if arguments.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arguments.device)
    model = torch.load(model_path, map_location=device, weights_only=False).eval()
    model.to(device)
    solver = GridSolver(model, device=device)

    gcmc_rows = read_gcmc_table(arguments.gcmc_table.expanduser().resolve())
    inputs = gyroid_inputs(arguments.gyroid_campaign.expanduser().resolve())
    rows = []
    for index, reference in enumerate(gcmc_rows, start=1):
        chemical_potential = float(reference["mu"])
        key = min(inputs, key=lambda value: abs(value - chemical_potential))
        if abs(key - chemical_potential) > 1.0e-8:
            raise ValueError("missing gyroid field at mu={}".format(chemical_potential))
        solved_path = fields_directory / "mu-{}-solved.extxyz".format(
            mu_label(chemical_potential)
        )
        result = solve_gyroid_state(
            inputs[key],
            solved_path,
            model,
            solver,
            device,
            arguments.maximum_density,
            arguments.max_iter,
            arguments.tolerance_residual,
            arguments.tolerance_rms_residual,
            arguments.tolerance_change,
        )
        result.update(
            {
                "mu_gcmc": chemical_potential,
                "temperature": float(reference["temperature"]),
                "rho_bulk_gcmc": float(reference["rho_bulk"]),
                "rho_bulk_sem": float(reference["rho_bulk_sem"]),
                "rho_gyroid_box_gcmc": float(reference["rho_gyroid_box"]),
                "rho_gyroid_box_sem": float(reference["rho_gyroid_box_sem"]),
            }
        )
        rows.append(result)
        print(
            "[{}/{}] mu={:+.1f}: converged={} iter={} density RMSE={:.6f}".format(
                index,
                len(gcmc_rows),
                chemical_potential,
                result["converged"],
                result["iterations"],
                result["density_rmse"],
            ),
            flush=True,
        )

    add_homogeneous_mu(
        rows,
        model,
        solver,
        device,
        arguments.bulk_box_length,
    )
    metrics = align_common_mu_gauge(rows)
    adsorption = build_adsorption_isotherm(rows)
    adsorption_reference = np.asarray(
        [row["rho_gyroid_box_gcmc"] for row in rows]
    )
    metrics["adsorption_isotherm"] = regression_metrics(
        adsorption_reference,
        adsorption["rho_gyroid_at_reference_bulk"],
    )

    temperature_label = "{:.3g}".format(rows[0]["temperature"])
    table_path = output_directory / "cdft_mu_density_T{}.csv".format(
        temperature_label
    )
    write_table(table_path, rows)
    write_adsorption_table(
        output_directory
        / "adsorption_isotherm_rho_T{}.csv".format(temperature_label),
        adsorption,
    )
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    run_configuration = {
        "model": str(model_path),
        "gyroid_campaign": str(
            arguments.gyroid_campaign.expanduser().resolve()
        ),
        "gcmc_table": str(arguments.gcmc_table.expanduser().resolve()),
        "device": str(device),
        "bulk_box_length": arguments.bulk_box_length,
        "maximum_density": arguments.maximum_density,
        "max_iter": arguments.max_iter,
        "tolerance_residual": arguments.tolerance_residual,
        "tolerance_rms_residual": arguments.tolerance_rms_residual,
        "tolerance_change": arguments.tolerance_change,
    }
    (output_directory / "run_config.json").write_text(
        json.dumps(run_configuration, indent=2) + "\n"
    )
    plot_cdft(
        output_directory / "cdft_mu_density_T{}.png".format(temperature_label),
        rows,
    )
    plot_comparison(
        output_directory
        / "gcmc_cdft_mu_density_T{}.png".format(temperature_label),
        rows,
    )
    plot_adsorption_isotherm(
        output_directory
        / "adsorption_isotherm_rho_T{}.png".format(temperature_label),
        rows,
        adsorption,
    )
    print("wrote {}".format(output_directory), flush=True)


if __name__ == "__main__":
    main()
