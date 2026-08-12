#!/usr/bin/env python3
"""Prepare a homogeneous-fluid GCMC scan at directly specified mu values."""

import argparse
import csv
import json
import shutil
from pathlib import Path


TEMPERATURE = 1.2
BOX_LENGTH = 8.0
CHEMICAL_POTENTIALS = tuple(-4.0 + 0.5 * index for index in range(11))

# Starting populations only shorten equilibration. The production density is
# determined by GCMC. These deliberately cover dilute through liquid-like
# initial states in the 8^3 box without assigning an EOS density to any mu.
INITIAL_PARTICLES = {
    -4.0: 25,
    -3.5: 50,
    -3.0: 150,
    -2.5: 300,
    -2.0: 360,
    -1.5: 380,
    -1.0: 400,
    -0.5: 410,
    0.0: 420,
    0.5: 430,
    1.0: 440,
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
                "system": "homogeneous",
                "temperature": TEMPERATURE,
                "chemical_potential": chemical_potential,
                "replica": 1,
                "initial_particles": INITIAL_PARTICLES[chemical_potential],
                "run_seed": 9382000 + 100 * task_id + 1,
                "input_file": "in.lj-homogeneous-gcmc",
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
        default=repository / "data" / "gyroid_adsorption" / "bulk_gcmc",
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
    input_template = Path(__file__).resolve().parent / "in.lj-homogeneous-gcmc"
    for row in rows:
        run_directory = campaign / row["run_directory"]
        run_directory.mkdir(parents=True, exist_ok=True)
        manifest.append(row["run_directory"])
        shutil.copy2(input_template, run_directory / row["input_file"])

        metadata = dict(row)
        metadata.update(
            {
                "box_length": BOX_LENGTH,
                "external_field": "none",
                "V_ext": 0.0,
                "gcmc_every": 50,
                "gcmc_exchanges": 20,
                "number_sample_every": 20,
                "nequil": 400000,
                "nsteps": 4000000,
            }
        )
        (run_directory / "statepoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    (campaign / "run_manifest.txt").write_text("\n".join(manifest) + "\n")
    print(
        "Prepared {} homogeneous direct-mu runs in {}.".format(
            len(rows),
            campaign,
        )
    )


if __name__ == "__main__":
    main()
