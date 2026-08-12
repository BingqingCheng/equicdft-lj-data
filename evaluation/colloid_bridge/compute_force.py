#!/usr/bin/env python3
"""Compute the two-colloid Hellmann--Feynman force from gridded densities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


APPLICATION_DIRECTORY = Path(__file__).resolve().parent
WORKSPACE = APPLICATION_DIRECTORY.parents[1]
DEFAULT_INPUT = APPLICATION_DIRECTORY / "in.lj-two-colloid-bridging.template"
DEFAULT_SOLVED_FIELD = WORKSPACE / "data" / "colloid_bridge" / "D7.0-solved.extxyz"


def read_lammps_parameters(path):
    """Read scalar ``variable NAME index VALUE`` entries from the input."""

    parameters = {}
    for line in path.read_text().splitlines():
        columns = line.split()
        if len(columns) == 4 and columns[0] == "variable" and columns[2] == "index":
            try:
                parameters[columns[1]] = float(columns[3])
            except ValueError:
                pass
    required = {
        "box_length",
        "separation",
        "colloid_radius",
        "core_height",
        "core_width",
        "shell_radius",
        "shell_width",
        "attraction_epsilon",
    }
    missing = required.difference(parameters)
    if missing:
        raise ValueError("missing LAMMPS variables: {}".format(sorted(missing)))
    return parameters


def minimum_image(displacement, box_length):
    return displacement - box_length * np.round(displacement / box_length)


def radial_field(distance, parameters):
    """Return the one-colloid potential and radial derivative d(phi)/dr."""

    core_radius = parameters["colloid_radius"]
    core_height = parameters["core_height"]
    core_width = parameters["core_width"]
    shell_radius = parameters["shell_radius"]
    shell_width = parameters["shell_width"]
    attraction = parameters["attraction_epsilon"]

    exponential = np.exp((distance - core_radius) / core_width)
    gaussian = np.exp(
        -(distance - shell_radius) ** 2 / (2.0 * shell_width**2)
    )
    potential = core_height / (1.0 + exponential) - attraction * gaussian
    derivative = (
        -core_height
        * exponential
        / (core_width * (1.0 + exponential) ** 2)
        + attraction
        * (distance - shell_radius)
        * gaussian
        / shell_width**2
    )
    return potential, derivative


def field_and_separation_derivative(positions, parameters, separation=None):
    """Return V_ext and its derivative with respect to symmetric separation."""

    box_length = parameters["box_length"]
    if separation is None:
        separation = parameters["separation"]
    centers = np.asarray(
        [
            [
                0.5 * box_length - 0.5 * separation,
                0.5 * box_length,
                0.5 * box_length,
            ],
            [
                0.5 * box_length + 0.5 * separation,
                0.5 * box_length,
                0.5 * box_length,
            ],
        ]
    )
    displacement_left = minimum_image(positions - centers[0], box_length)
    displacement_right = minimum_image(positions - centers[1], box_length)
    distance_left = np.linalg.norm(displacement_left, axis=1)
    distance_right = np.linalg.norm(displacement_right, axis=1)
    potential_left, derivative_left = radial_field(distance_left, parameters)
    potential_right, derivative_right = radial_field(distance_right, parameters)

    # R_L,x = L/2-D/2 and R_R,x = L/2+D/2, hence
    # dr_L/dD = +s_L,x/(2r_L) and dr_R/dD = -s_R,x/(2r_R).
    dfield_dseparation = 0.5 * (
        derivative_left * displacement_left[:, 0] / distance_left
        - derivative_right * displacement_right[:, 0] / distance_right
    )
    return potential_left + potential_right, dfield_dseparation


def force_from_density(density, derivative, cell_volume):
    """Return F_D = -DeltaV sum rho_g dV_g/dD."""

    return float(-cell_volume * np.sum(density * derivative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--solved-field", type=Path, default=DEFAULT_SOLVED_FIELD)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--finite-difference-step", type=float, default=1.0e-4)
    arguments = parser.parse_args()

    lammps_input = arguments.lammps_input.expanduser().resolve()
    solved_path = arguments.solved_field.expanduser().resolve()
    output_directory = arguments.output_directory
    if output_directory is None:
        output_directory = solved_path.parent
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    parameters = read_lammps_parameters(lammps_input)
    atoms = read(solved_path, index=0)
    positions = np.asarray(atoms.arrays["grid_center"], dtype=float)
    rho_md = np.asarray(atoms.arrays["reference_density"], dtype=float)
    rho_cdft = np.asarray(atoms.arrays["density"], dtype=float)
    stored_field = np.asarray(atoms.arrays["V_ext"], dtype=float)
    grid_spacing = np.asarray(atoms.info["grid_spacing"], dtype=float)
    cell_volume = float(np.prod(grid_spacing))

    reconstructed_field, derivative = field_and_separation_derivative(
        positions,
        parameters,
    )
    potential_cap = float(atoms.info.get("V_ext_cap", np.inf))
    comparison_field = np.minimum(reconstructed_field, potential_cap)
    field_error = comparison_field - stored_field

    delta = arguments.finite_difference_step
    field_plus, _ = field_and_separation_derivative(
        positions,
        parameters,
        parameters["separation"] + delta,
    )
    field_minus, _ = field_and_separation_derivative(
        positions,
        parameters,
        parameters["separation"] - delta,
    )
    derivative_fd = (field_plus - field_minus) / (2.0 * delta)
    derivative_error = derivative - derivative_fd

    force_md = force_from_density(rho_md, derivative, cell_volume)
    force_cdft = force_from_density(rho_cdft, derivative, cell_volume)
    particle_number_md = float(cell_volume * np.sum(rho_md))
    particle_number_cdft = float(cell_volume * np.sum(rho_cdft))

    result = {
        "separation": parameters["separation"],
        "temperature": float(atoms.info["T"]),
        "cell_volume": cell_volume,
        "particle_number_md": particle_number_md,
        "particle_number_cdft": particle_number_cdft,
        "F_D_from_MD_density": force_md,
        "F_D_from_cDFT_density": force_cdft,
        "cDFT_minus_MD_force": force_cdft - force_md,
        "relative_force_difference": (
            (force_cdft - force_md) / force_md if force_md != 0.0 else None
        ),
        "field_reconstruction_max_abs_error": float(
            np.max(np.abs(field_error))
        ),
        "field_reconstruction_rms_error": float(
            np.sqrt(np.mean(field_error**2))
        ),
        "derivative_finite_difference_max_abs_error": float(
            np.max(np.abs(derivative_error))
        ),
        "derivative_finite_difference_rms_error": float(
            np.sqrt(np.mean(derivative_error**2))
        ),
        "finite_difference_step": delta,
        "force_units": "epsilon/sigma",
        "sign_convention": "F_D < 0 attractive; F_D > 0 repulsive",
        "lammps_input": str(lammps_input),
        "solved_field": str(solved_path),
    }

    stem = output_directory / "two_colloid_hellmann_feynman_force_D7"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    annotated = atoms.copy()
    annotated.set_array("dVext_dD", derivative)
    annotated.set_array("hf_force_density_MD", -rho_md * derivative)
    annotated.set_array("hf_force_density_cDFT", -rho_cdft * derivative)
    annotated.info.update(
        {
            "F_D_from_MD_density": force_md,
            "F_D_from_cDFT_density": force_cdft,
            "hf_force_units": "epsilon/sigma",
        }
    )
    write(stem.with_suffix(".extxyz"), annotated, format="extxyz")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
