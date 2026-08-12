#!/usr/bin/env python3
"""Evaluate fixed-N equilibrium solves on temperature-separated NVT fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


from equicdft import GridData, GridSolver  # noqa: E402


def select_indices(count, fraction):
    """Return an evenly spaced deterministic fraction of one T file."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    selected_count = max(1, int(round(fraction * count)))
    if selected_count == 1:
        return [count // 2]
    return [
        int(round(index * (count - 1) / (selected_count - 1)))
        for index in range(selected_count)
    ]


def empty_stats():
    return {
        "frames": 0,
        "converged": 0,
        "iterations": 0,
        "evaluations": 0,
        "points": 0,
        "sse": 0.0,
        "target_sum": 0.0,
        "target_square_sum": 0.0,
        "converged_points": 0,
        "converged_sse": 0.0,
        "converged_target_sum": 0.0,
        "converged_target_square_sum": 0.0,
    }


def add_frame(stats, target, prediction, converged, iterations, evaluations):
    error = prediction - target
    count = int(target.numel())
    sse = float(error.square().sum())
    target_sum = float(target.sum())
    target_square_sum = float(target.square().sum())
    stats["frames"] += 1
    stats["converged"] += int(converged)
    stats["iterations"] += int(iterations)
    stats["evaluations"] += int(evaluations)
    stats["points"] += count
    stats["sse"] += sse
    stats["target_sum"] += target_sum
    stats["target_square_sum"] += target_square_sum
    if converged:
        stats["converged_points"] += count
        stats["converged_sse"] += sse
        stats["converged_target_sum"] += target_sum
        stats["converged_target_square_sum"] += target_square_sum


def rmse_r2(sse, count, target_sum, target_square_sum):
    if count == 0:
        return math.nan, math.nan
    rmse = math.sqrt(sse / count)
    centered = target_square_sum - target_sum * target_sum / count
    r2 = math.nan if centered <= 0.0 else 1.0 - sse / centered
    return rmse, r2


def summarize(model, group, stats, settings):
    rmse, r2 = rmse_r2(
        stats["sse"],
        stats["points"],
        stats["target_sum"],
        stats["target_square_sum"],
    )
    converged_rmse, converged_r2 = rmse_r2(
        stats["converged_sse"],
        stats["converged_points"],
        stats["converged_target_sum"],
        stats["converged_target_square_sum"],
    )
    return {
        "model": model,
        "group": group,
        "solve_frames": stats["frames"],
        "solve_converged": stats["converged"],
        "solve_mean_iterations": stats["iterations"] / stats["frames"],
        "solve_mean_evaluations": stats["evaluations"] / stats["frames"],
        "vext_to_rho_rmse_all": rmse,
        "vext_to_rho_r2_all": r2,
        "vext_to_rho_rmse_converged": converged_rmse,
        "vext_to_rho_r2_converged": converged_r2,
        **settings,
    }


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, default=repository / "model")
    parser.add_argument("--data-directory", type=Path, default=repository / "data" / "benchmarks" / "large_nvt")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository / "output" / "benchmarks",
    )
    parser.add_argument("--output-prefix", default="nvt_large_solve")
    parser.add_argument("--method", choices=("euler", "minimize"), default="euler")
    parser.add_argument("--beta-multiplier", type=float, default=0.0)
    parser.add_argument("--maximum-density", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tolerance-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-6)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_directory = arguments.fit_directory.expanduser().resolve()
    model = torch.load(
        fit_directory / "model.pt",
        map_location=device,
        weights_only=False,
    ).eval()
    solver = GridSolver(model, device=device)
    paths = sorted(arguments.data_directory.expanduser().resolve().glob("*.extxyz"))
    if not paths:
        raise FileNotFoundError("no top-level EXTXYZ files found")

    accumulators = defaultdict(empty_stats)
    frame_rows = []
    for path in paths:
        frames = GridData.from_xyz(path, grid_info=model.grid_info)
        for frame_index in select_indices(len(frames), arguments.fraction):
            frame = frames[frame_index]
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in frame.items()
            }
            target = moved.pop("rho")
            temperature = float(moved["temperature"].detach().cpu())
            cell_volume = torch.prod(moved["grid_spacing"])
            particle_numbers = cell_volume * torch.sum(target, dim=0)
            solved = solver.solve(
                moved,
                particle_numbers=particle_numbers,
                method=arguments.method,
                beta_multiplier=arguments.beta_multiplier,
                max_iter=arguments.max_iter,
                tolerance_residual=arguments.tolerance_residual,
                tolerance_change=arguments.tolerance_change,
                residual_density_threshold=model.rho_min,
                maximum_density=arguments.maximum_density,
            )
            prediction = solved["rho"].detach()
            converged = bool(solved["converged"])
            groups = ("all", "T={:.1f}".format(temperature))
            for group in groups:
                add_frame(
                    accumulators[group],
                    target,
                    prediction,
                    converged,
                    solved["n_iter"],
                    solved["n_evaluations"],
                )

            error = prediction - target
            centered = float(
                target.square().sum()
                - target.sum().square() / target.numel()
            )
            sse = float(error.square().sum())
            frame_rows.append(
                {
                    "model": fit_directory.name,
                    "source_file": path.name,
                    "source_frame": frame_index,
                    "temperature": temperature,
                    "particle_number": float(particle_numbers.sum()),
                    "mean_density": float(target.mean()),
                    "density_variance": float(target.var(unbiased=False)),
                    "converged": converged,
                    "iterations": int(solved["n_iter"]),
                    "evaluations": int(solved["n_evaluations"]),
                    "rho_rmse": math.sqrt(sse / target.numel()),
                    "rho_r2": math.nan if centered <= 0.0 else 1.0 - sse / centered,
                }
            )

    settings = {
        "solver_method": arguments.method,
        "beta_multiplier": arguments.beta_multiplier,
        "maximum_density": arguments.maximum_density,
        "max_iter": arguments.max_iter,
        "tolerance_residual": arguments.tolerance_residual,
        "tolerance_change": arguments.tolerance_change,
    }
    metric_rows = [
        summarize(fit_directory.name, group, stats, settings)
        for group, stats in accumulators.items()
    ]
    metric_rows.sort(key=lambda row: (row["group"] != "all", row["group"]))
    prefix = arguments.output_prefix
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / (prefix + "_metrics.json")).write_text(
        json.dumps(metric_rows, indent=2, allow_nan=True) + "\n"
    )
    with (output_directory / (prefix + "_frames.csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    overall = metric_rows[0]
    print(
        "{}: RMSE={:.6f} R2={:.6f} converged={}/{}".format(
            fit_directory.name,
            overall["vext_to_rho_rmse_all"],
            overall["vext_to_rho_r2_all"],
            overall["solve_converged"],
            overall["solve_frames"],
        )
    )


if __name__ == "__main__":
    main()
