#!/usr/bin/env python3
"""Postprocess one gyroid-adsorption or bulk-validation GCMC run."""

import argparse
import json
import math
import sys
from pathlib import Path

# ``collect_density_external.py`` is shared by all six application geometries
# and therefore remains one directory above this gyroid-specific workflow.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colloid_bridge"))
from collect_reference_density import (
    grid_geometry,
    make_evaluator,
    read_density_grid,
)


def read_number_series(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = [float(value) for value in line.split()]
        if len(columns) != 7:
            raise ValueError(
                "expected seven columns in {}, found {}".format(path, len(columns))
            )
        rows.append(columns)
    if not rows:
        raise ValueError("no number samples in {}".format(path))
    # fix ave/time writes the production configuration at reset step zero.
    # Density averaging begins at the first positive sampling step, so remove
    # the zero-step row to keep number and density averages exactly aligned.
    if len(rows) > 1 and rows[0][0] == 0.0:
        rows = rows[1:]
    return rows


def mean(values):
    return sum(values) / float(len(values))


def summarize_numbers(rows, system, box_length):
    particle_numbers = [row[1] for row in rows]
    mean_number = mean(particle_numbers)
    number_variance = mean(
        [(number - mean_number) ** 2 for number in particle_numbers]
    )
    last = rows[-1]
    summary = {
        "n_samples": len(rows),
        "mean_particle_number": mean_number,
        "mean_box_density": mean_number / box_length ** 3,
        "particle_number_variance": number_variance,
        "S0_from_number_fluctuations": (
            number_variance / mean_number if mean_number > 0.0 else 0.0
        ),
        "insertion_attempts": int(round(last[3])),
        "insertion_successes": int(round(last[4])),
        "deletion_attempts": int(round(last[5])),
        "deletion_successes": int(round(last[6])),
    }
    summary["insertion_acceptance"] = (
        summary["insertion_successes"] / float(summary["insertion_attempts"])
        if summary["insertion_attempts"]
        else 0.0
    )
    summary["deletion_acceptance"] = (
        summary["deletion_successes"] / float(summary["deletion_attempts"])
        if summary["deletion_attempts"]
        else 0.0
    )

    if system == "gyroid":
        pore_numbers = [row[2] for row in rows]
        mean_pore = mean(pore_numbers)
        covariance = mean(
            [
                (pore - mean_pore) * (number - mean_number)
                for pore, number in zip(pore_numbers, particle_numbers)
            ]
        )
        summary["mean_pore_particle_number"] = mean_pore
        summary["pore_susceptibility"] = covariance

    return summary


def write_gyroid_grid(run_directory, statepoint, summary):
    density_rows = read_density_grid(run_directory / "density_grid.dat")
    _axes, shape, spacing, lengths, index_maps = grid_geometry(density_rows, 0.5)
    parameters = {
        "g0": statepoint["g0"],
        "wall_height": statepoint["wall_height"],
        "wall_width": statepoint["wall_width"],
    }
    evaluate = make_evaluator("gyroid-pore", parameters, lengths)

    combined = []
    density_integral = 0.0
    for bin_id, x, y, z, density in density_rows:
        potential = evaluate(x, y, z)
        combined.append((bin_id, x, y, z, density, potential))
        density_integral += density * math.prod(spacing)

    external_lines = [
        "# Raw positive logistic gyroid potential; no additive shift",
        "# Bin x y z V_ext",
    ]
    for bin_id, x, y, z, _density, potential in combined:
        external_lines.append(
            "{} {:.16g} {:.16g} {:.16g} {:.16g}".format(
                bin_id, x, y, z, potential
            )
        )
    (run_directory / "external_potential_grid.dat").write_text(
        "\n".join(external_lines) + "\n"
    )

    nx, ny, nz = shape
    lx, ly, lz = lengths
    comment = (
        'Lattice="{} 0 0 0 {} 0 0 0 {}" '.format(nx, ny, nz)
        + 'physical_lattice="{:.16g} 0 0 0 {:.16g} 0 0 0 {:.16g}" '.format(
            lx, ly, lz
        )
        + "Properties=species:S:1:pos:R:3:grid_center:R:3:grid_id:I:1:"
        + "density:R:1:V_ext:R:1 "
        + 'pbc="T T T" T={:.16g} mu={:.16g} '.format(
            statepoint["temperature"], statepoint["chemical_potential"]
        )
        + 'grid_size="{} {} {}" '.format(nx, ny, nz)
        + 'grid_spacing="{:.16g} {:.16g} {:.16g}" '.format(*spacing)
        + 'grid_indexing="zero_based" ensemble=GCMC '
        + "N_mean={:.16g} N_pore_mean={:.16g} ".format(
            summary["mean_particle_number"],
            summary["mean_pore_particle_number"],
        )
        + 'application="gyroid-pore-adsorption" '
        + "g0={:.16g} wall_height={:.16g} wall_width={:.16g} ".format(
            statepoint["g0"],
            statepoint["wall_height"],
            statepoint["wall_width"],
        )
        + 'V_ext_gauge="raw_positive_logistic"'
    )
    output_lines = [str(len(combined)), comment]
    for bin_id, x, y, z, density, potential in combined:
        output_lines.append(
            "X {} {} {} {:.16g} {:.16g} {:.16g} {} {:.16g} {:.16g}".format(
                index_maps[0][x],
                index_maps[1][y],
                index_maps[2][z],
                x,
                y,
                z,
                bin_id,
                density,
                potential,
            )
        )
    output = run_directory / "density_external_grid_0.5sigma.extxyz"
    output.write_text("\n".join(output_lines) + "\n")
    summary["density_integral"] = density_integral
    summary["density_number_difference"] = (
        density_integral - summary["mean_particle_number"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()

    run_directory = args.run_directory.resolve()
    statepoint = json.loads((run_directory / "statepoint.json").read_text())
    number_rows = read_number_series(
        run_directory / "particle_number_timeseries.dat"
    )
    summary = summarize_numbers(
        number_rows,
        statepoint["system"],
        statepoint["box_length"],
    )
    summary.update(
        {
            "system": statepoint["system"],
            "temperature": statepoint["temperature"],
            "chemical_potential": statepoint["chemical_potential"],
            "replica": statepoint["replica"],
        }
    )
    if statepoint.get("target_bulk_density") is not None:
        summary["target_bulk_density"] = statepoint["target_bulk_density"]
    if statepoint.get("reference_pressure") is not None:
        summary["reference_pressure"] = statepoint["reference_pressure"]

    if statepoint["system"] == "gyroid":
        write_gyroid_grid(run_directory, statepoint, summary)

    (run_directory / "gcmc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        "{}: <N>={:.6f}, insertion acceptance={:.6g}, "
        "deletion acceptance={:.6g}".format(
            run_directory,
            summary["mean_particle_number"],
            summary["insertion_acceptance"],
            summary["deletion_acceptance"],
        )
    )


if __name__ == "__main__":
    main()
