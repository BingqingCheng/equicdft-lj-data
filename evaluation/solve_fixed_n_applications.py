#!/usr/bin/env python3
"""Solve the six LJ application fields with this fitted functional."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write


REPOSITORY = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = REPOSITORY / "model"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def regression_metrics(reference: np.ndarray, prediction: np.ndarray):
    error = prediction - reference
    rmse = float(np.sqrt(np.mean(error**2)))
    variance = float(np.mean((reference - np.mean(reference)) ** 2))
    r2 = float("nan") if variance == 0.0 else 1.0 - float(np.mean(error**2)) / variance
    return rmse, r2


def application_inputs(root: Path, selected=None):
    paths = sorted(root.glob("*/density_external_grid_0.5sigma.extxyz"))
    if selected:
        selected = set(selected)
        paths = [
            path
            for path in paths
            if path.parent.name in selected
            or path.parent.name.split("-", 1)[-1] in selected
        ]
    if not paths:
        raise FileNotFoundError("no application EXTXYZ files found in {}".format(root))
    return paths


def solve_application(
    input_path,
    output_path,
    model,
    solver,
    device,
    max_iter,
    maximum_density,
    tolerance_residual,
    tolerance_change,
):
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

    cell_volume = torch.prod(solve_data["grid_spacing"])
    particle_numbers = cell_volume * torch.sum(reference_rho, dim=0)
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

    reference = reference_rho[:, 0].detach().cpu().numpy()
    prediction = solved["rho"][:, 0].detach().cpu().numpy()
    grid_size = np.asarray(atoms.info["grid_size"], dtype=int)
    grid_positions = np.rint(atoms.get_positions()).astype(int)
    if str(atoms.info.get("grid_indexing", "zero_based")).lower().startswith("one"):
        grid_positions -= 1
    flat_positions = np.ravel_multi_index(grid_positions.T, tuple(grid_size), order="C")
    reference_in_atoms_order = reference[flat_positions]
    prediction_in_atoms_order = prediction[flat_positions]

    atoms.set_array("reference_density", reference_in_atoms_order)
    atoms.set_array("density", prediction_in_atoms_order)
    atoms.set_array(
        "density_error",
        prediction_in_atoms_order - reference_in_atoms_order,
    )
    atoms.info.update(
        {
            "model": MODEL_DIRECTORY.name,
            "density": "solved fixed-N equilibrium density",
            "reference_density": "MD application density",
            "solve_method": "adaptive_euler_fixed_N",
            "solve_initialization": "uniform beta_multiplier=0",
            "solve_converged": bool(solved["converged"]),
            "solve_iterations": int(solved["n_iter"]),
            "solve_evaluations": int(solved["n_evaluations"]),
            "solve_max_iter": max_iter,
            "solve_max_residual": scalar(solved["max_euler_lagrange_residual"]),
            "solve_rms_residual": scalar(solved["rms_euler_lagrange_residual"]),
            "solve_tolerance_residual": tolerance_residual,
            "solve_tolerance_change": tolerance_change,
            "solve_maximum_density": maximum_density,
        }
    )
    write(output_path, atoms, format="extxyz")

    rmse, r2 = regression_metrics(reference, prediction)
    return {
        "application": str(atoms.info.get("application", input_path.parent.name)),
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "temperature": float(frame["temperature"].detach().cpu()),
        "particle_number": float(particle_numbers[0].detach().cpu()),
        "n_grid": int(reference.size),
        "converged": bool(solved["converged"]),
        "iterations": int(solved["n_iter"]),
        "evaluations": int(solved["n_evaluations"]),
        "max_residual": scalar(solved["max_euler_lagrange_residual"]),
        "rms_residual": scalar(solved["rms_euler_lagrange_residual"]),
        "density_rmse": rmse,
        "density_r2": r2,
        "predicted_density_min": float(np.min(prediction)),
        "predicted_density_max": float(np.max(prediction)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--applications-root",
        type=Path,
        default=REPOSITORY / "data" / "fixed_n_applications",
    )
    parser.add_argument(
        "--application",
        action="append",
        help="solve only this application folder or application name; repeatable",
    )
    parser.add_argument("--model", type=Path, default=MODEL_DIRECTORY / "model.pt")
    parser.add_argument("--output-directory", type=Path, default=REPOSITORY / "output" / "fixed_n_applications")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--maximum-density", type=float, default=2.0)
    parser.add_argument("--tolerance-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-6)
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="replace selected rows in an existing solve_summary.csv",
    )
    arguments = parser.parse_args()

    from equicdft import GridSolver

    device = torch.device(arguments.device)
    model = torch.load(
        arguments.model.expanduser().resolve(),
        map_location=device,
        weights_only=False,
    ).eval()
    model.to(device)
    solver = GridSolver(model, device=device)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for input_path in application_inputs(
        arguments.applications_root.expanduser().resolve(),
        arguments.application,
    ):
        application = input_path.parent.name.split("-", 1)[-1]
        output_path = arguments.output_directory / "{}-solved.extxyz".format(application)
        row = solve_application(
            input_path=input_path,
            output_path=output_path,
            model=model,
            solver=solver,
            device=device,
            max_iter=arguments.max_iter,
            maximum_density=arguments.maximum_density,
            tolerance_residual=arguments.tolerance_residual,
            tolerance_change=arguments.tolerance_change,
        )
        rows.append(row)
        print(
            "{application}: converged={converged} iterations={iterations} "
            "RMSE={density_rmse:.6g} R2={density_r2:.6g}".format(**row),
            flush=True,
        )

    summary_path = arguments.output_directory / "solve_summary.csv"
    if arguments.update_summary and summary_path.is_file():
        with summary_path.open(newline="") as handle:
            previous_rows = list(csv.DictReader(handle))
        updated_applications = {row["application"] for row in rows}
        rows = [
            row
            for row in previous_rows
            if row["application"] not in updated_applications
        ] + rows
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
