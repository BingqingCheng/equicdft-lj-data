#!/usr/bin/env python3
"""Prepare the ten fixed-N soft two-colloid separation states."""

import argparse
import json
import shutil
from pathlib import Path


SEPARATIONS = (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0)
SEEDS = {
    4.5: (8121211, 8121212),
    5.0: (8121221, 8121222),
    5.5: (8121011, 8121012),
    6.0: (8121021, 8121022),
    6.5: (8121031, 8121032),
    7.0: (8121041, 8121042),
    7.5: (8121051, 8121052),
    8.0: (8121061, 8121062),
    9.0: (8121071, 8121072),
    10.0: (8121231, 8121232),
}


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository / "data" / "colloid_bridge" / "reference_scan",
    )
    arguments = parser.parse_args()
    campaign_root = arguments.output_directory.expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    template_path = Path(__file__).resolve().parent / "in.lj-two-colloid-bridging.template"
    template = template_path.read_text()
    shutil.copy2(
        Path(__file__).resolve().parent / "collect_reference_density.py",
        campaign_root / "collect_density_external.py",
    )

    manifest = []
    statepoints = []
    for separation in SEPARATIONS:
        run_name = "D{:.1f}".format(separation)
        run_directory = campaign_root / run_name
        run_directory.mkdir(exist_ok=True)

        placement_seed, velocity_seed = SEEDS[separation]
        rendered = template.replace("@SEPARATION@", "{:.1f}".format(separation))
        rendered = rendered.replace("@PLACEMENT_SEED@", str(placement_seed))
        rendered = rendered.replace("@VELOCITY_SEED@", str(velocity_seed))
        if "@" in rendered:
            raise ValueError("unresolved template placeholder for {}".format(run_name))
        (run_directory / "in.lj-two-colloid-bridging").write_text(rendered)

        statepoint = {
            "run": run_name,
            "application": "two-colloid-bridging",
            "ensemble": "NVT",
            "temperature": 1.2,
            "particle_count": 4600,
            "box_length": 20.0,
            "separation": separation,
            "colloid_radius": 2.6,
            "core_height": 8.0,
            "core_width": 0.5,
            "shell_radius": 3.4,
            "shell_width": 0.8,
            "attraction_epsilon": 1.4,
            "grid_spacing": 0.5,
            "force_sample_every": 10,
            "equilibration_steps": 400000,
            "production_steps": 4000000,
            "placement_seed": placement_seed,
            "velocity_seed": velocity_seed,
        }
        (run_directory / "statepoint.json").write_text(
            json.dumps(statepoint, indent=2) + "\n"
        )
        manifest.append(run_name)
        statepoints.append(statepoint)

    (campaign_root / "run_manifest.txt").write_text("\n".join(manifest) + "\n")
    (campaign_root / "statepoints.json").write_text(
        json.dumps(statepoints, indent=2) + "\n"
    )
    print("Prepared {} separation states in {}".format(len(manifest), campaign_root))


if __name__ == "__main__":
    main()
