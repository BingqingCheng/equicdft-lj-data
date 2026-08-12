#!/usr/bin/env python3
"""Prepare the direct-chemical-potential gyroid GCMC scan."""

import argparse
import csv
import json
import shutil
from pathlib import Path


TEMPERATURE = 1.2
BOX_LENGTH = 16.0
CHEMICAL_POTENTIALS = tuple(-4.0 + 0.5 * index for index in range(11))

# These values only provide reasonably close starting populations. GCMC
# determines the equilibrium particle number; the initial count is not used as
# a thermodynamic label or target.
INITIAL_PARTICLES = {
    -4.0: 100,
    -3.5: 200,
    -3.0: 500,
    -2.5: 1000,
    -2.0: 1200,
    -1.5: 1200,
    -1.0: 1200,
    -0.5: 1200,
    0.0: 1200,
    0.5: 1200,
    1.0: 1200,
}


def safe_number(value):
    """Return a compact path-safe representation of a floating-point value."""

    return ("{:.1f}".format(value)).replace("-", "m").replace(".", "p")


def build_rows():
    rows = []
    for task_id, chemical_potential in enumerate(CHEMICAL_POTENTIALS, start=1):
        run_directory = "mu-{}/run-001".format(safe_number(chemical_potential))
        rows.append(
            {
                "task_id": task_id,
                "system": "gyroid",
                "temperature": TEMPERATURE,
                "chemical_potential": chemical_potential,
                "replica": 1,
                "initial_particles": INITIAL_PARTICLES[chemical_potential],
                "run_seed": 9282000 + 100 * task_id + 1,
                "input_file": "in.lj-gyroid-gcmc",
                "run_directory": run_directory,
            }
        )
    return rows


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "data" / "gyroid_adsorption" / "gyroid_gcmc",
    )
    args = parser.parse_args()

    campaign = args.output.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    columns = tuple(rows[0].keys())

    with (campaign / "statepoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = []
    input_template = Path(__file__).resolve().parent / "in.lj-gyroid-gcmc"
    for row in rows:
        run_directory = campaign / row["run_directory"]
        run_directory.mkdir(parents=True, exist_ok=True)
        manifest.append(row["run_directory"])
        shutil.copy2(input_template, run_directory / row["input_file"])

        metadata = dict(row)
        metadata.update(
            {
                "box_length": BOX_LENGTH,
                "grid_spacing": 0.5,
                "g0": 0.0,
                "wall_height": 8.0,
                "wall_width": 0.35,
                "V_ext_gauge": "raw_positive_logistic",
                "gcmc_every": 50,
                "gcmc_exchanges": 20,
                "density_sample_every": 20,
                "nequil": 400000,
                "nsteps": 4000000,
            }
        )
        (run_directory / "statepoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    (campaign / "run_manifest.txt").write_text("\n".join(manifest) + "\n")
    print(
        "Prepared {} direct-mu gyroid runs in {}.".format(
            len(rows),
            campaign,
        )
    )


if __name__ == "__main__":
    main()
