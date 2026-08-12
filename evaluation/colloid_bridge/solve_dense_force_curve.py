#!/usr/bin/env python3
"""Compute a dense fixed-N cDFT force curve for the two-colloid field.

No MD density field is needed.  For every requested separation, the script
constructs the regular grid and analytic external potential, solves the cDFT
Euler--Lagrange equation at fixed particle number, and evaluates

    F_D = -DeltaV sum_g rho_g dV_ext,g/dD.

Separations are solved from large to small so each converged density can seed
the neighboring state point.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


SCAN_DIRECTORY = Path(__file__).resolve().parent
WORKSPACE = SCAN_DIRECTORY.parents[1]
DEFAULT_MODEL_DIRECTORY = WORKSPACE / "model"
DEFAULT_OUTPUT_DIRECTORY = WORKSPACE / "output" / "colloid_bridge"
DEFAULT_LAMMPS_INPUT = WORKSPACE / "data" / "colloid_bridge" / "reference_scan" / "D7.0" / "in.lj-two-colloid-bridging"

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


def separation_grid(minimum, maximum, step, solve_order):
    """Return an inclusive grid without accumulated roundoff."""

    if np.isclose(minimum, maximum):
        return np.asarray([float(maximum)])
    count = int(round((maximum - minimum) / step))
    if solve_order == "descending":
        values = maximum - step * np.arange(count + 1, dtype=float)
        endpoint = minimum
    else:
        values = minimum + step * np.arange(count + 1, dtype=float)
        endpoint = maximum
    if not np.isclose(values[-1], endpoint):
        raise ValueError("the separation range must be an integer number of steps")
    return values


def to_device(data, device):
    """Move tensor-valued grid data to the selected device."""

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def write_rows(path, rows):
    """Write sorted results after each state point."""

    ordered = sorted(rows, key=lambda row: row["separation"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY / "model.pt",
    )
    parser.add_argument("--lammps-input", type=Path, default=DEFAULT_LAMMPS_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-separation", type=float, default=4.5)
    parser.add_argument("--maximum-separation", type=float, default=10.0)
    parser.add_argument("--separation-step", type=float, default=0.1)
    parser.add_argument(
        "--solve-order",
        choices=("descending", "ascending"),
        default="descending",
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--maximum-density", type=float, default=2.0)
    parser.add_argument("--tolerance-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-rms-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-6)
    arguments = parser.parse_args()

    if arguments.separation_step <= 0.0:
        raise ValueError("separation-step must be positive")
    if arguments.minimum_separation > arguments.maximum_separation:
        raise ValueError("minimum-separation must not exceed maximum-separation")

    from equicdft import GridData, GridSolver

    device = torch.device(arguments.device)
    model_path = arguments.model.expanduser().resolve()
    model = torch.load(model_path, map_location=device, weights_only=False).eval()
    model.to(device)
    solver = GridSolver(model, device=device)

    parameters = read_lammps_parameters(arguments.lammps_input.expanduser().resolve())
    temperature = parameters["T"]
    particle_number = parameters["particle_count"]
    box_length = parameters["box_length"]
    spacing = np.asarray(model.grid_info["grid_spacing"], dtype=float)
    grid_size_float = box_length / spacing
    grid_size = np.rint(grid_size_float).astype(int)
    if not np.allclose(grid_size, grid_size_float):
        raise ValueError("box length is not commensurate with the model grid")

    data = GridData.from_dict(
        {
            "grid_size": grid_size.tolist(),
            "grid_spacing": spacing.tolist(),
            "temperature": temperature,
            "n_types": int(model.grid_info["n_types"]),
        },
        grid_info=model.grid_info,
    )
    data = to_device(data, device)
    dtype = next(model.parameters()).dtype

    # GridData stores integer lattice indices; the external field is sampled
    # at the physical center of each finite-volume cell.
    grid_indices = data["grid_positions"].detach().cpu().numpy()
    grid_centers = (grid_indices + 0.5) * spacing[None, :]
    cell_volume = float(np.prod(spacing))

    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "cdft_force_dense.csv"

    rows = []
    initial_rho = None
    for separation in separation_grid(
        arguments.minimum_separation,
        arguments.maximum_separation,
        arguments.separation_step,
        arguments.solve_order,
    ):
        external_field, field_derivative = field_and_separation_derivative(
            grid_centers,
            parameters,
            separation=float(separation),
        )
        solve_data = dict(data)
        solve_data["V_ext"] = torch.as_tensor(
            external_field[:, None],
            dtype=dtype,
            device=device,
        )
        solve_arguments = dict(
            particle_numbers=[particle_number],
            method="euler",
            beta_multiplier=0.0,
            max_iter=arguments.max_iter,
            tolerance_residual=arguments.tolerance_residual,
            tolerance_rms_residual=arguments.tolerance_rms_residual,
            tolerance_change=arguments.tolerance_change,
            residual_density_threshold=float(model.rho_min),
            maximum_density=arguments.maximum_density,
        )
        solved = solver.solve(
            solve_data,
            initial_rho=initial_rho,
            **solve_arguments,
        )
        # A neighboring density is normally the fastest initialization, but a
        # rare state may land on a poor fixed-point branch.  Retry such states
        # from the uniform profile and keep the result with the smaller maximum
        # Euler--Lagrange residual.
        retried_from_uniform = False
        if initial_rho is not None and not solved["converged"]:
            retry = solver.solve(
                solve_data,
                initial_rho=None,
                **solve_arguments,
            )
            if (
                retry["converged"]
                or scalar(retry["max_euler_lagrange_residual"])
                < scalar(solved["max_euler_lagrange_residual"])
            ):
                solved = retry
                retried_from_uniform = True
        initial_rho = solved["rho"].detach()
        density = initial_rho[:, 0].cpu().numpy()
        row = {
            "separation": float(separation),
            "HF_force_cDFT_density": force_from_density(
                density,
                field_derivative,
                cell_volume,
            ),
            "average_chemical_potential": scalar(
                solved["average_chemical_potential"][0]
            )
            / scalar(data["beta"]),
            "particle_number": cell_volume * float(np.sum(density)),
            "solve_converged": bool(solved["converged"]),
            "solve_iterations": int(solved["n_iter"]),
            "solve_evaluations": int(solved["n_evaluations"]),
            "retried_from_uniform": retried_from_uniform,
            "solve_max_residual": scalar(
                solved["max_euler_lagrange_residual"]
            ),
            "solve_rms_residual": scalar(
                solved["rms_euler_lagrange_residual"]
            ),
            "predicted_density_min": float(np.min(density)),
            "predicted_density_max": float(np.max(density)),
        }
        rows.append(row)
        write_rows(output_path, rows)
        print(
            "D={separation:.2f} converged={solve_converged} "
            "iterations={solve_iterations} F={HF_force_cDFT_density:+.6f}".format(
                **row
            ),
            flush=True,
        )

    run_config = {
        "model": str(model_path),
        "lammps_input": str(arguments.lammps_input.expanduser().resolve()),
        "device": str(device),
        "temperature": temperature,
        "particle_number": particle_number,
        "box_length": box_length,
        "grid_size": grid_size.tolist(),
        "grid_spacing": spacing.tolist(),
        "minimum_separation": arguments.minimum_separation,
        "maximum_separation": arguments.maximum_separation,
        "separation_step": arguments.separation_step,
        "solve_order": "{} separation with density continuation".format(
            arguments.solve_order
        ),
        "max_iter": arguments.max_iter,
        "maximum_density": arguments.maximum_density,
        "tolerance_residual": arguments.tolerance_residual,
        "tolerance_rms_residual": arguments.tolerance_rms_residual,
        "tolerance_change": arguments.tolerance_change,
        "force_definition": "F_D = -DeltaV sum_g rho_g dVext_g/dD",
        "force_sign": "negative attractive; positive repulsive",
    }
    (output_directory / "cdft_force_dense_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
