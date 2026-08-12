#!/usr/bin/env python3
"""Solve fixed-N liquid-vapor slab profiles for a family of fitted models."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[2]
TEMPERATURES = (0.65, 0.80, 0.95, 1.05)
OUTPUT_STEM = "interface_density_T0.65_0.80_0.95_1.05"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def temperature_label(temperature: float) -> str:
    return "{:.2f}".format(temperature).rstrip("0").rstrip(".")


def planar_profile(atoms, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Average a grid field over the two directions parallel to the interface."""

    z = np.asarray(atoms.arrays["grid_center"][:, 2], dtype=float)
    values = np.asarray(atoms.arrays[key], dtype=float)
    unique_z, inverse = np.unique(np.round(z, decimals=12), return_inverse=True)
    counts = np.bincount(inverse)
    average = np.bincount(inverse, weights=values) / counts
    return unique_z, average


def regression_metrics(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    rmse = float(np.sqrt(np.mean(error**2)))
    centered = np.asarray(reference) - float(np.mean(reference))
    denominator = float(np.sum(centered**2))
    r2 = float("nan") if denominator == 0.0 else 1.0 - float(np.sum(error**2)) / denominator
    return rmse, r2


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def solve_model(
    model_folder: Path,
    slab_directory: Path,
    temperatures: tuple[float, ...],
    device_name: str,
    max_iter: int,
    maximum_density: float,
    tolerance_residual: float,
    tolerance_rms_residual: float,
    tolerance_change: float,
    skip_existing: bool,
) -> tuple[str, list[dict[str, object]]]:
    """Load one model, solve all requested temperatures, and write its outputs."""

    from equicdft import GridData, GridSolver

    device = torch.device(device_name)
    model_path = model_folder / "model.pt"
    model = torch.load(model_path, map_location=device, weights_only=False).eval()
    model.to(device)
    solver = GridSolver(model, device=device)
    output_folder = ROOT / "output" / "bulk_response" / "interfaces" / model_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)
    rows = []

    for temperature in temperatures:
        label = temperature_label(temperature)
        input_path = slab_directory / (
            f"nvt-slab-L6x6x24-T{label}-0.5sigma.extxyz"
        )
        output_path = output_folder / (
            f"nvt-slab-L6x6x24-T{label}-solved.extxyz"
        )
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        if skip_existing and output_path.is_file():
            atoms = read(output_path)
            reference = np.asarray(atoms.arrays["reference_density"], dtype=float)
            prediction = np.asarray(atoms.arrays["density"], dtype=float)
            converged = bool(atoms.info.get("solve_converged", False))
            iterations = int(atoms.info.get("solve_iterations", 0))
            max_residual = float(atoms.info.get("solve_max_residual", np.nan))
            rms_residual = float(atoms.info.get("solve_rms_residual", np.nan))
        else:
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
            if "V_ext" not in solve_data:
                solve_data["V_ext"] = torch.zeros_like(reference_rho)
            cell_volume = torch.prod(solve_data["grid_spacing"])
            particle_numbers = cell_volume * torch.sum(reference_rho, dim=0)

            solved = solver.solve(
                solve_data,
                initial_rho=torch.clamp(reference_rho, min=1.0e-8),
                particle_numbers=particle_numbers,
                method="euler",
                max_iter=max_iter,
                tolerance_residual=tolerance_residual,
                tolerance_rms_residual=tolerance_rms_residual,
                tolerance_change=tolerance_change,
                mixing=0.05,
                adaptive_mixing=True,
                minimum_mixing=0.005,
                maximum_mixing=0.2,
                mixing_growth=1.1,
                mixing_backtrack_factor=0.5,
                max_log_density_change=2.0,
                residual_density_threshold=float(getattr(model, "rho_min", 1.0e-3)),
                maximum_density=maximum_density,
            )

            reference = reference_rho[:, 0].detach().cpu().numpy()
            prediction = solved["rho"][:, 0].detach().cpu().numpy()
            converged = bool(solved["converged"])
            iterations = int(solved["n_iter"])
            max_residual = scalar(solved["max_euler_lagrange_residual"])
            rms_residual = scalar(solved["rms_euler_lagrange_residual"])

            atoms.set_array("reference_density", reference)
            atoms.set_array("density", prediction)
            atoms.set_array("density_error", prediction - reference)
            atoms.info.update(
                {
                    "model": model_folder.name,
                    "density": "solved fixed-N equilibrium density",
                    "reference_density": "MD slab density",
                    "solve_method": "adaptive_euler_fixed_N",
                    "solve_converged": converged,
                    "solve_iterations": iterations,
                    "solve_max_residual": max_residual,
                    "solve_rms_residual": rms_residual,
                    "solve_tolerance_residual": tolerance_residual,
                    "solve_tolerance_rms_residual": tolerance_rms_residual,
                    "solve_tolerance_change": tolerance_change,
                    "solve_maximum_density": maximum_density,
                }
            )
            write(output_path, atoms, format="extxyz")

        _z, reference_profile = planar_profile(atoms, "reference_density")
        _z, prediction_profile = planar_profile(atoms, "density")
        grid_rmse, grid_r2 = regression_metrics(reference, prediction)
        profile_rmse, profile_r2 = regression_metrics(
            reference_profile,
            prediction_profile,
        )
        row = {
            "temperature": temperature,
            "converged": converged,
            "iterations": iterations,
            "max_residual": max_residual,
            "rms_residual": rms_residual,
            "grid_rmse": grid_rmse,
            "grid_r2": grid_r2,
            "profile_rmse": profile_rmse,
            "profile_r2": profile_r2,
            "minimum_density": float(np.min(prediction)),
            "maximum_density": float(np.max(prediction)),
        }
        rows.append(row)
        print(
            f"{model_folder.name} T={label} converged={converged} "
            f"iterations={iterations} profile_RMSE={profile_rmse:.5f}",
            flush=True,
        )

    summary_path = output_folder / "interface_solve_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_model(output_folder, model_folder.name, temperatures)
    return model_folder.name, rows


def plot_model(output_folder: Path, model_name: str, temperatures: tuple[float, ...]) -> None:
    """Plot the planar MD and solved profiles for one fitted model."""

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), sharex=True)
    for axis, temperature in zip(axes.ravel(), temperatures):
        label = temperature_label(temperature)
        atoms = read(
            output_folder
            / f"nvt-slab-L6x6x24-T{label}-solved.extxyz"
        )
        z, reference = planar_profile(atoms, "reference_density")
        _z, prediction = planar_profile(atoms, "density")
        profile_rmse, _profile_r2 = regression_metrics(reference, prediction)
        axis.plot(z, reference, color="black", linewidth=2.0, label="MD reference")
        axis.plot(z, prediction, color="#0072B2", linewidth=1.8, label="cDFT equilibrium")
        converged = bool(atoms.info.get("solve_converged", False))
        iterations = int(atoms.info.get("solve_iterations", 0))
        status = "converged" if converged else "not converged"
        axis.text(
            0.98,
            0.95,
            f"{status}, {iterations} iterations\nRMSE = {profile_rmse:.4f}",
            ha="right",
            va="top",
            transform=axis.transAxes,
            color="0.35",
            fontsize=8.5,
        )
        axis.set_title(rf"$T^*={temperature:.2f}$")
        axis.set_xlim(0.0, 24.0)
        ymax = max(float(np.max(reference)), float(np.max(prediction)))
        axis.set_ylim(0.0, max(0.1, 1.08 * ymax))
        axis.grid(alpha=0.18)

    axes.ravel()[0].legend(frameon=False, loc="upper left")
    figure.suptitle(model_name)
    figure.supxlabel(r"$z/\sigma$")
    figure.supylabel(r"number density $\rho\sigma^3$")
    figure.tight_layout()
    figure.savefig(output_folder / f"{OUTPUT_STEM}.png", dpi=260)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=ROOT,
    )
    parser.add_argument(
        "--slab-directory",
        type=Path,
        default=ROOT / "data" / "bulk_response" / "slabs",
    )
    parser.add_argument("--models", nargs="+", default=("model",))
    parser.add_argument("--temperatures", nargs=4, type=float, default=TEMPERATURES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--maximum-density", type=float, default=2.0)
    parser.add_argument("--tolerance-residual", type=float, default=3.0e-2)
    parser.add_argument("--tolerance-rms-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-5)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_model_subprocess(model_folder: Path, args: argparse.Namespace) -> str:
    """Run one model in an independent process for safe parallel execution."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.root.expanduser().resolve()),
        "--slab-directory",
        str(args.slab_directory.expanduser().resolve()),
        "--models",
        model_folder.name,
        "--temperatures",
        *(str(value) for value in args.temperatures),
        "--workers",
        "1",
        "--device",
        args.device,
        "--max-iter",
        str(args.max_iter),
        "--maximum-density",
        str(args.maximum_density),
        "--tolerance-residual",
        str(args.tolerance_residual),
        "--tolerance-rms-residual",
        str(args.tolerance_rms_residual),
        "--tolerance-change",
        str(args.tolerance_change),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-interface")
    environment.setdefault("XDG_CACHE_HOME", "/private/tmp")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout.strip()


def collect_summaries(root: Path, model_folders: list[Path]) -> Path:
    """Collect per-model diagnostics into one campaign-level CSV file."""

    rows = []
    for model_folder in model_folders:
        path = ROOT / "output" / "bulk_response" / "interfaces" / model_folder.name / "interface_solve_summary.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"model": model_folder.name, **row})
    output = ROOT / "output" / "bulk_response" / "interfaces" / "interface_solve_summary_all_models.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    slab_directory = args.slab_directory.expanduser().resolve()
    if args.models:
        model_folders = [root / name for name in args.models]
    else:
        model_folders = sorted(
            path for path in root.iterdir() if (path / "model.pt").is_file()
        )
    if not model_folders:
        raise FileNotFoundError(f"No model.pt files found under {root}")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("parallel model solving currently requires --device cpu")

    job_arguments = (
        slab_directory,
        tuple(args.temperatures),
        args.device,
        args.max_iter,
        args.maximum_density,
        args.tolerance_residual,
        args.tolerance_rms_residual,
        args.tolerance_change,
        args.skip_existing,
    )
    if args.workers == 1:
        for model_folder in model_folders:
            solve_model(model_folder, *job_arguments)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_model_subprocess, model_folder, args): model_folder
                for model_folder in model_folders
            }
            for future in as_completed(futures):
                output = future.result()
                if output:
                    print(output, flush=True)
                print(f"completed {futures[future].name}", flush=True)
    completed_model_folders = model_folders
    summary_path = collect_summaries(root, completed_model_folders)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
