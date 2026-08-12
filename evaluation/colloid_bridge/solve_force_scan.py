#!/usr/bin/env python3
"""Solve the separation scan and evaluate Hellmann--Feynman forces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write


SCAN_DIRECTORY = Path(__file__).resolve().parent
WORKSPACE = SCAN_DIRECTORY.parents[1]
REFERENCE_DIRECTORY = WORKSPACE / "data" / "colloid_bridge" / "reference_scan"
DEFAULT_MODEL_DIRECTORY = WORKSPACE / "model"
DEFAULT_OUTPUT_DIRECTORY = WORKSPACE / "output" / "colloid_bridge"

from compute_force import (  # noqa: E402
    field_and_separation_derivative,
    force_from_density,
    read_lammps_parameters,
)


def scalar(value):
    """Convert a scalar tensor or number to a Python float."""

    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def regression_metrics(reference, prediction):
    """Return grid-point RMSE and R2 for a solved density field."""

    error = prediction - reference
    mean_squared_error = float(np.mean(error**2))
    variance = float(np.mean((reference - np.mean(reference)) ** 2))
    r2 = float("nan") if variance == 0.0 else 1.0 - mean_squared_error / variance
    return float(np.sqrt(mean_squared_error)), r2


def separation_direct_forces(path):
    """Read the direct MD mean-force measurements keyed by separation."""

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        float(row["separation"]): {
            "mean": float(row["mean_force"]),
            "sem": float(row["block_sem"]),
        }
        for row in rows
    }


def input_paths(directory, selected=None):
    """Return completed fields in increasing separation order."""

    paths = list(directory.glob("D*/density_external_grid_0.5sigma.extxyz"))
    paths.sort(key=lambda path: float(path.parent.name[1:]))
    if not paths:
        raise ValueError("no completed separation-scan fields were found")
    if selected:
        selected = set(selected)
        paths = [
            path
            for path in paths
            if float(path.parent.name[1:]) in selected
        ]
        if len(paths) != len(selected):
            raise ValueError("one or more requested separations are unavailable")
    return paths


def atoms_order(values, atoms):
    """Map the model's regular-grid ordering back to the EXTXYZ row ordering."""

    grid_size = np.asarray(atoms.info["grid_size"], dtype=int)
    grid_positions = np.rint(atoms.get_positions()).astype(int)
    if str(atoms.info.get("grid_indexing", "zero_based")).lower().startswith("one"):
        grid_positions -= 1
    flat_positions = np.ravel_multi_index(
        grid_positions.T,
        tuple(grid_size),
        order="C",
    )
    return values[flat_positions]


def write_summary(path, rows):
    """Write accumulated results so completed separations survive interruption."""

    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solve_and_evaluate(
    input_path,
    output_path,
    model,
    solver,
    device,
    direct_force,
    max_iter,
    maximum_density,
    tolerance_residual,
    tolerance_change,
):
    """Solve one fixed-N field and evaluate both density-based HF forces."""

    from equicdft import GridData

    atoms = read(input_path)
    frame = GridData.from_xyz(input_path, grid_info=model.grid_info)[0]
    frame = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in frame.items()
    }
    reference_rho = frame["rho"]
    solve_data = dict(frame)
    solve_data.pop("rho")
    solve_data.pop("mu", None)
    solve_data.pop("beta_mu", None)

    cell_volume_tensor = torch.prod(solve_data["grid_spacing"])
    particle_numbers = cell_volume_tensor * torch.sum(reference_rho, dim=0)
    solved = solver.solve(
        solve_data,
        particle_numbers=particle_numbers,
        method="euler",
        beta_multiplier=0.0,
        max_iter=max_iter,
        tolerance_residual=tolerance_residual,
        tolerance_change=tolerance_change,
        residual_density_threshold=float(model.rho_min),
        maximum_density=maximum_density,
    )

    reference_grid = reference_rho[:, 0].detach().cpu().numpy()
    prediction_grid = solved["rho"][:, 0].detach().cpu().numpy()
    reference = atoms_order(reference_grid, atoms)
    prediction = atoms_order(prediction_grid, atoms)

    lammps_input = input_path.parent / "in.lj-two-colloid-bridging"
    parameters = read_lammps_parameters(lammps_input)
    positions = np.asarray(atoms.arrays["grid_center"], dtype=float)
    reconstructed_field, derivative = field_and_separation_derivative(
        positions,
        parameters,
    )
    stored_field = np.asarray(atoms.arrays["V_ext"], dtype=float)
    potential_cap = float(atoms.info.get("V_ext_cap", np.inf))
    field_error = np.minimum(reconstructed_field, potential_cap) - stored_field

    grid_spacing = np.asarray(atoms.info["grid_spacing"], dtype=float)
    cell_volume = float(np.prod(grid_spacing))
    force_hf_md = force_from_density(reference, derivative, cell_volume)
    force_hf_cdft = force_from_density(prediction, derivative, cell_volume)
    density_rmse, density_r2 = regression_metrics(reference, prediction)

    atoms.set_array("reference_density", reference)
    atoms.set_array("density", prediction)
    atoms.set_array("density_error", prediction - reference)
    atoms.set_array("dVext_dD", derivative)
    atoms.set_array("hf_force_density_MD", -reference * derivative)
    atoms.set_array("hf_force_density_cDFT", -prediction * derivative)
    atoms.info.update(
        {
            "model": DEFAULT_MODEL_DIRECTORY.name,
            "density": "solved fixed-N equilibrium density",
            "reference_density": "MD separation-scan density",
            "solve_method": "adaptive_euler_fixed_N",
            "solve_initialization": "uniform beta_multiplier=0",
            "solve_converged": bool(solved["converged"]),
            "solve_iterations": int(solved["n_iter"]),
            "solve_evaluations": int(solved["n_evaluations"]),
            "solve_max_iter": max_iter,
            "solve_max_residual": scalar(
                solved["max_euler_lagrange_residual"]
            ),
            "solve_rms_residual": scalar(
                solved["rms_euler_lagrange_residual"]
            ),
            "solve_tolerance_residual": tolerance_residual,
            "solve_tolerance_change": tolerance_change,
            "solve_maximum_density": maximum_density,
            "F_D_direct_MD": direct_force["mean"],
            "F_D_HF_MD_density": force_hf_md,
            "F_D_HF_cDFT_density": force_hf_cdft,
            "hf_force_units": "epsilon/sigma",
            "hf_force_sign": "negative attractive; positive repulsive",
        }
    )
    write(output_path, atoms, format="extxyz")

    particle_number_md = cell_volume * float(np.sum(reference))
    particle_number_cdft = cell_volume * float(np.sum(prediction))
    return {
        "separation": parameters["separation"],
        "direct_MD_force": direct_force["mean"],
        "direct_MD_block_sem": direct_force["sem"],
        "HF_force_MD_density": force_hf_md,
        "HF_force_cDFT_density": force_hf_cdft,
        "HF_MD_minus_direct": force_hf_md - direct_force["mean"],
        "HF_cDFT_minus_direct": force_hf_cdft - direct_force["mean"],
        "HF_cDFT_minus_HF_MD": force_hf_cdft - force_hf_md,
        "density_RMSE": density_rmse,
        "density_R2": density_r2,
        "particle_number_MD": particle_number_md,
        "particle_number_cDFT": particle_number_cdft,
        "solve_converged": bool(solved["converged"]),
        "solve_max_iter": max_iter,
        "solve_iterations": int(solved["n_iter"]),
        "solve_evaluations": int(solved["n_evaluations"]),
        "solve_max_residual": scalar(solved["max_euler_lagrange_residual"]),
        "solve_rms_residual": scalar(solved["rms_euler_lagrange_residual"]),
        "predicted_density_min": float(np.min(prediction)),
        "predicted_density_max": float(np.max(prediction)),
        "field_reconstruction_max_abs_error": float(
            np.max(np.abs(field_error))
        ),
        "output": str(output_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY / "model.pt",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--reference-directory", type=Path, default=REFERENCE_DIRECTORY
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--separation",
        type=float,
        action="append",
        help="solve only this separation; repeatable",
    )
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--maximum-density", type=float, default=2.0)
    parser.add_argument("--tolerance-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-6)
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="replace selected separations in an existing summary",
    )
    arguments = parser.parse_args()

    from equicdft import GridSolver

    reference_directory = arguments.reference_directory.expanduser().resolve()

    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "hellmann_feynman_force_summary.csv"
    direct_forces = separation_direct_forces(
        reference_directory / "mean_force_summary.csv"
    )

    device = torch.device(arguments.device)
    model_path = arguments.model.expanduser().resolve()
    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    ).eval()
    model.to(device)
    solver = GridSolver(model, device=device)

    rows_by_separation = {}
    if arguments.update_summary and summary_path.is_file():
        with summary_path.open(newline="") as handle:
            for previous in csv.DictReader(handle):
                previous.setdefault("solve_max_iter", "200")
                rows_by_separation[float(previous["separation"])] = previous

    for input_path in input_paths(reference_directory, arguments.separation):
        separation = float(input_path.parent.name[1:])
        output_path = output_directory / "D{:.1f}-solved.extxyz".format(separation)
        row = solve_and_evaluate(
            input_path=input_path,
            output_path=output_path,
            model=model,
            solver=solver,
            device=device,
            direct_force=direct_forces[separation],
            max_iter=arguments.max_iter,
            maximum_density=arguments.maximum_density,
            tolerance_residual=arguments.tolerance_residual,
            tolerance_change=arguments.tolerance_change,
        )
        rows_by_separation[separation] = row
        write_summary(
            summary_path,
            [rows_by_separation[key] for key in sorted(rows_by_separation)],
        )
        print(
            "D={separation:.1f}: converged={solve_converged} "
            "iterations={solve_iterations} RMSE={density_RMSE:.6f} "
            "F_direct={direct_MD_force:+.6f} F_HF_MD={HF_force_MD_density:+.6f} "
            "F_HF_cDFT={HF_force_cDFT_density:+.6f}".format(**row),
            flush=True,
        )

    config_path = output_directory / "run_config.json"
    if arguments.update_summary and config_path.is_file():
        run_config = json.loads(config_path.read_text())
        run_config.setdefault("updates", []).append(
            {
                "separations": arguments.separation,
                "max_iter": arguments.max_iter,
                "tolerance_residual": arguments.tolerance_residual,
                "tolerance_change": arguments.tolerance_change,
            }
        )
    else:
        run_config = {
            "model": str(model_path),
            "device": str(device),
            "max_iter": arguments.max_iter,
            "maximum_density": arguments.maximum_density,
            "tolerance_residual": arguments.tolerance_residual,
            "tolerance_change": arguments.tolerance_change,
            "initialization": "uniform beta_multiplier=0",
            "ensemble": "fixed N",
            "force_definition": "F_D = -DeltaV sum_g rho_g dVext_g/dD",
            "force_sign": "negative attractive; positive repulsive",
        }
    config_path.write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    print("Wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
