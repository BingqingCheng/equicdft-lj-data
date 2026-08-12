#!/usr/bin/env python3
"""Evaluate one comprehensive-NVT fit on the three held-out temperatures."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


from equicdft import GridData, GridSolver  # noqa: E402


def empty_stats():
    return {
        "frames": 0,
        "forward_points": 0,
        "forward_sse": 0.0,
        "forward_target_sum": 0.0,
        "forward_target_square_sum": 0.0,
        "constancy_weight": 0.0,
        "constancy_sse": 0.0,
        "solve_frames": 0,
        "solve_converged": 0,
        "solve_iterations": 0,
        "solve_evaluations": 0,
        "density_points": 0,
        "density_sse": 0.0,
        "density_target_sum": 0.0,
        "density_target_square_sum": 0.0,
    }


def group_names(temperature, family):
    temperature_name = "T={:.1f}".format(temperature)
    return ("all", temperature_name, family, temperature_name + ":" + family)


def load_test_set(manifest_path, roots, grid_info):
    """Load complete held-out source files in manifest order."""

    manifest = json.loads(manifest_path.read_text())
    paths = []
    metadata = []
    for record in manifest["sources"]:
        if record["partition"] != "test":
            continue
        path = roots[record["collection"]] / record["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        metadata.extend(
            {
                "temperature": float(record["temperature"]),
                "field_family": record["field_family"],
            }
            for _ in range(int(record["frames"]))
        )

    frames = GridData.from_xyz(paths, grid_info=grid_info)
    if len(frames) != len(metadata):
        raise RuntimeError("manifest and loaded test frame counts differ")
    return frames, metadata


def selected_solve_indices(metadata, fraction):
    """Select an evenly spaced fraction within every T/family group."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("solve_fraction must lie in (0, 1]")
    groups = defaultdict(list)
    for index, record in enumerate(metadata):
        groups[(record["temperature"], record["field_family"])].append(index)

    selected = set()
    for indices in groups.values():
        count = max(1, int(round(fraction * len(indices))))
        if count == 1:
            selected.add(indices[len(indices) // 2])
        else:
            selected.update(
                indices[
                    int(round(position * (len(indices) - 1) / (count - 1)))
                ]
                for position in range(count)
            )
    return selected


def add_forward(stats, frame, result, model):
    """Accumulate gauge-aligned rho -> beta*V_ext errors."""

    rho = result["rho"].detach()
    target = frame["beta"].to(rho) * frame["V_ext"].to(rho)
    wavelength = model.thermal_wavelength.to(rho)
    raw_prediction = result["c1"].detach() - torch.log(
        torch.clamp(
            rho * wavelength.pow(3),
            min=torch.finfo(rho.dtype).tiny,
        )
    )
    mask = rho > model.rho_min
    denominator = mask.sum(dim=0).clamp_min(1)
    offset = ((target - raw_prediction) * mask).sum(dim=0) / denominator
    prediction = raw_prediction + offset
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    error = selected_prediction - selected_target

    local_mu = result["local_chemical_potential"].detach()
    average_mu = result["average_chemical_potential"].detach()
    weights = result["chemical_potential_weights"].detach()
    stats["frames"] += 1
    stats["forward_points"] += int(selected_target.numel())
    stats["forward_sse"] += float(error.square().sum())
    stats["forward_target_sum"] += float(selected_target.sum())
    stats["forward_target_square_sum"] += float(selected_target.square().sum())
    stats["constancy_weight"] += float(weights.sum())
    stats["constancy_sse"] += float(
        ((local_mu - average_mu).square() * weights).sum()
    )


def add_solve(
    stats,
    prediction,
    target,
    converged,
    iterations,
    evaluations,
):
    target = target.detach()
    error = prediction - target
    stats["solve_frames"] += 1
    stats["solve_converged"] += int(converged)
    stats["solve_iterations"] += int(iterations)
    stats["solve_evaluations"] += int(evaluations)
    stats["density_points"] += int(target.numel())
    stats["density_sse"] += float(error.square().sum())
    stats["density_target_sum"] += float(target.sum())
    stats["density_target_square_sum"] += float(target.square().sum())


def rmse_r2(sse, count, target_sum, target_square_sum):
    if count == 0:
        return math.nan, math.nan
    rmse = math.sqrt(sse / count)
    centered = target_square_sum - target_sum * target_sum / count
    r2 = math.nan if centered <= 0.0 else 1.0 - sse / centered
    return rmse, r2


def summarize(model_name, group, stats):
    forward_rmse, forward_r2 = rmse_r2(
        stats["forward_sse"],
        stats["forward_points"],
        stats["forward_target_sum"],
        stats["forward_target_square_sum"],
    )
    density_rmse, density_r2 = rmse_r2(
        stats["density_sse"],
        stats["density_points"],
        stats["density_target_sum"],
        stats["density_target_square_sum"],
    )
    constancy_rmse = (
        math.sqrt(stats["constancy_sse"] / stats["constancy_weight"])
        if stats["constancy_weight"] > 0.0
        else math.nan
    )
    return {
        "model": model_name,
        "group": group,
        "frames": stats["frames"],
        "rho_to_beta_vext_rmse": forward_rmse,
        "rho_to_beta_vext_r2": forward_r2,
        "local_mu_constancy_rmse": constancy_rmse,
        "solve_frames": stats["solve_frames"],
        "solve_converged": stats["solve_converged"],
        "solve_mean_iterations": (
            stats["solve_iterations"] / stats["solve_frames"]
            if stats["solve_frames"]
            else math.nan
        ),
        "solve_mean_evaluations": (
            stats["solve_evaluations"] / stats["solve_frames"]
            if stats["solve_frames"]
            else math.nan
        ),
        "vext_to_rho_rmse": density_rmse,
        "vext_to_rho_r2": density_r2,
    }


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, default=repository / "model")
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "output" / "benchmarks" / "heldout_nvt_metrics.json",
    )
    parser.add_argument("--manifest", type=Path, default=repository / "manifests" / "split_manifest.json")
    parser.add_argument("--general-directory", type=Path, default=repository / "data" / "training" / "general")
    parser.add_argument("--dimensional-directory", type=Path, default=repository / "data" / "training" / "dimensional")
    parser.add_argument("--mixed-directory", type=Path, default=repository / "data" / "training" / "mixed")
    parser.add_argument("--extended-directory", type=Path, default=repository / "data" / "training" / "extended")
    parser.add_argument("--solve-fraction", type=float, default=1.0)
    parser.add_argument(
        "--method",
        choices=("euler", "minimize"),
        default="euler",
    )
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
    roots = {
        "general": arguments.general_directory.expanduser().resolve(),
        "dimensional": arguments.dimensional_directory.expanduser().resolve(),
        "mixed": arguments.mixed_directory.expanduser().resolve(),
        "extended": arguments.extended_directory.expanduser().resolve(),
    }
    frames, metadata = load_test_set(
        arguments.manifest.expanduser().resolve(),
        roots,
        model.grid_info,
    )
    solve_indices = selected_solve_indices(metadata, arguments.solve_fraction)
    accumulators = defaultdict(empty_stats)

    for index, (frame, record) in enumerate(zip(frames, metadata)):
        groups = group_names(record["temperature"], record["field_family"])
        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in frame.items()
        }
        result = solver.evaluate(moved, compute_c1=True)
        for group in groups:
            add_forward(accumulators[group], moved, result, model)

        if index in solve_indices:
            solve_data = dict(moved)
            target_density = solve_data.pop("rho")
            cell_volume = torch.prod(solve_data["grid_spacing"])
            particle_numbers = cell_volume * torch.sum(target_density, dim=0)
            solved = solver.solve(
                solve_data,
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
            for group in groups:
                add_solve(
                    accumulators[group],
                    prediction,
                    target_density,
                    bool(solved["converged"]),
                    solved["n_iter"],
                    solved["n_evaluations"],
                )

    rows = [
        summarize(fit_directory.name, group, stats)
        for group, stats in accumulators.items()
    ]
    rows.sort(key=lambda row: (row["group"] != "all", row["group"]))
    for row in rows:
        row["solver_method"] = arguments.method
        row["beta_multiplier"] = arguments.beta_multiplier
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")
    overall = rows[0]
    print(
        "{}: forward={:.5f} solve={:.5f} converged={}/{}".format(
            fit_directory.name,
            overall["rho_to_beta_vext_rmse"],
            overall["vext_to_rho_rmse"],
            overall["solve_converged"],
            overall["solve_frames"],
        )
    )


if __name__ == "__main__":
    main()
