#!/usr/bin/env python3
"""Train one SR model on the expanded L=8, 0.5-sigma NVT split."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset


REPOSITORY = Path(os.environ["EQUICDFT_REPOSITORY"]).expanduser().resolve()
sys.path.insert(0, str(REPOSITORY / "src"))

from equicdft import (  # noqa: E402
    CartesianAFeatures,
    CartesianBFeatures,
    GridCACEModel,
    GridData,
    LDAReadout,
    LocalReadout,
    Loss,
    Metrics,
    TensorLoss,
    Trainer,
    make_dataloaders,
)


def resolve_sources(manifest, roots):
    development_paths = []
    train_indices = []
    valid_indices = []
    offset = 0
    for record in manifest["sources"]:
        if record["partition"] != "development":
            continue
        path = roots[record["collection"]] / record["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        development_paths.append(path)
        local_valid = set(record["valid_frame_indices"])
        valid_indices.extend(offset + index for index in sorted(local_valid))
        train_indices.extend(
            offset + index
            for index in range(record["frames"])
            if index not in local_valid
        )
        offset += record["frames"]
    return development_paths, train_indices, valid_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--general-directory", type=Path, required=True)
    parser.add_argument("--dimensional-directory", type=Path, required=True)
    parser.add_argument("--mixed-directory", type=Path, required=True)
    parser.add_argument("--extended-directory", type=Path, required=True)
    parser.add_argument("--mixed-more-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-power", type=int, choices=(2, 3), required=True)
    parser.add_argument("--cutoff-grid", type=int, choices=(3, 4), required=True)
    parser.add_argument(
        "--radial-basis",
        choices=("none", "gaussian"),
        default="none",
    )
    parser.add_argument("--radial-exponents", type=float, nargs="+")
    parser.add_argument(
        "--trainable-radial-exponents",
        action="store_true",
    )
    parser.add_argument(
        "--coordinate-scaling",
        choices=("none", "cutoff"),
        default="none",
    )
    parser.add_argument(
        "--free-energy-mode",
        choices=("physical", "beta"),
        required=True,
    )
    parser.add_argument(
        "--fit-name",
        help="Explicit output directory name for an exploratory configuration.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=200)
    arguments = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(arguments.manifest.read_text())
    roots = {
        "general": arguments.general_directory.expanduser().resolve(),
        "dimensional": arguments.dimensional_directory.expanduser().resolve(),
        "mixed": arguments.mixed_directory.expanduser().resolve(),
        "extended": arguments.extended_directory.expanduser().resolve(),
        "mixed_more": arguments.mixed_more_directory.expanduser().resolve(),
    }
    source_paths, train_indices, valid_indices = resolve_sources(manifest, roots)

    output_directory = arguments.output_directory.expanduser().resolve()
    fit_name = arguments.fit_name
    if fit_name is None:
        fit_name = "fit-{}-p{}nu2-sr-c{}".format(
            arguments.free_energy_mode,
            arguments.max_power,
            arguments.cutoff_grid,
        )
    if Path(fit_name).name != fit_name:
        raise ValueError("fit-name must be a directory name, not a path")
    fit_directory = output_directory / fit_name
    checkpoint_directory = fit_directory / "checkpoints"
    log_directory = fit_directory / "logs"
    fit_directory.mkdir(parents=True, exist_ok=True)

    development_dataset = GridData.from_xyz(
        source_paths,
        cutoff_grid=arguments.cutoff_grid,
        boltzmann_constant=1.0,
        thermal_wavelength=1.0,
    )
    if len(development_dataset) != len(train_indices) + len(valid_indices):
        raise RuntimeError("manifest and loaded development frame counts differ")
    train_dataset = Subset(development_dataset, train_indices)
    valid_dataset = Subset(development_dataset, valid_indices)
    loaders = make_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        compute_mean_density=True,
        compute_mean_temperature=True,
    )
    n_types = int(development_dataset[0]["n_types"].item())

    a_features = CartesianAFeatures(
        mean_density=loaders["mean_density"],
        cutoff_grid=arguments.cutoff_grid,
        max_power=arguments.max_power,
        radial_basis=arguments.radial_basis,
        radial_exponents=arguments.radial_exponents,
        trainable_radial_exponents=arguments.trainable_radial_exponents,
        coordinate_scaling=arguments.coordinate_scaling,
        separate_center=True,
        n_types=n_types,
    )
    b_features = CartesianBFeatures(
        max_power=arguments.max_power,
        max_product_order=2,
    )
    readouts = [
        LDAReadout(
            mean_density=loaders["mean_density"],
            n_types=n_types,
            hidden_sizes=(32, 16),
        ),
        LocalReadout(n_types=n_types, hidden_sizes=(32, 16)),
    ]
    rho_min = 1.0e-3
    model = GridCACEModel(
        a_features=a_features,
        b_features=b_features,
        readout=readouts,
        grid_spacing=development_dataset[0]["grid_spacing"],
        mean_temperature=loaders["mean_temperature"],
        boltzmann_constant=1.0,
        thermal_wavelength=1.0,
        compute_c1=True,
        compute_local_mu=True,
        rho_min=rho_min,
        free_energy_mode=arguments.free_energy_mode,
    ).to(device)

    loss = Loss(
        terms=[
            TensorLoss(
                name="local_chemical_potential",
                prediction_key="local_chemical_potential",
                target_key="average_chemical_potential",
                weights_key="chemical_potential_weights",
            )
        ]
    )
    metrics = [
        Metrics(
            target_key="average_chemical_potential",
            prediction_key="local_chemical_potential",
            name="local chemical potential",
            metric_keys=("mae", "rmse", "rmse_percent", "r2"),
            selection_mask_key="chemical_potential_weights",
        )
    ]
    trainer = Trainer(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": 1.0e-4},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={"factor": 0.5, "patience": 3, "min_lr": 1.0e-6},
        device=device,
        checkpoint_dir=checkpoint_directory,
        checkpoint_interval=5,
        save_best=True,
        early_stopping_patience=arguments.early_stopping_patience,
        log_dir=log_directory,
    )

    completed_epochs = 0
    last_checkpoint = checkpoint_directory / "last.pt"
    if last_checkpoint.is_file():
        completed_epochs = trainer.load_checkpoint(
            last_checkpoint,
            train_loader=loaders["train"],
        )
    remaining_epochs = max(0, arguments.epochs - completed_epochs)
    if remaining_epochs:
        trainer.fit(
            train_loader=loaders["train"],
            valid_loader=loaders["valid"],
            epochs=remaining_epochs,
        )

    best = torch.load(
        checkpoint_directory / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    torch.save(model, fit_directory / "model.pt")

    run_config = {
        "seed": arguments.seed,
        "epoch_budget": arguments.epochs,
        "completed_epochs": len(trainer.history),
        "best_epoch": int(best["record"]["epoch"]),
        "best_valid_loss": float(best["record"]["valid_losses"]["total"]),
        "batch_size": arguments.batch_size,
        "cutoff_grid": arguments.cutoff_grid,
        "max_power": arguments.max_power,
        "max_product_order": 2,
        "free_energy_mode": arguments.free_energy_mode,
        "rho_min": rho_min,
        "radial_basis": arguments.radial_basis,
        "radial_exponents": arguments.radial_exponents,
        "trainable_radial_exponents": (
            arguments.trainable_radial_exponents
        ),
        "coordinate_scaling": arguments.coordinate_scaling,
        "separate_center": True,
        "use_lda": True,
        "long_range_kernel": None,
        "mean_density": float(loaders["mean_density"]),
        "mean_temperature": float(loaders["mean_temperature"]),
        "training_target": "spatially constant local chemical potential",
        "train_frames": len(train_dataset),
        "valid_frames": len(valid_dataset),
        "test_frames": manifest["totals"]["test"],
        "split_manifest": str(arguments.manifest.resolve()),
        "code_commit": os.environ.get("EQUICDFT_COMMIT"),
        "device": str(device),
    }
    (fit_directory / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    shutil.copy2(Path(__file__), fit_directory / "training.py")


if __name__ == "__main__":
    main()
