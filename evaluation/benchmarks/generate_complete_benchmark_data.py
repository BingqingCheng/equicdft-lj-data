#!/usr/bin/env python3
"""Evaluate every small- and large-system NVT test field for the paper figure."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


TEMPERATURES = (0.7, 1.1, 1.5)
RANDOM_SEED = 20260812


def parse_arguments():
    script = Path(__file__).resolve()
    project = script.parents[2]
    default_fit = project / "model"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, default=default_fit)
    parser.add_argument(
        "--output-directory", type=Path, default=project / "output" / "benchmarks"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project / "manifests" / "split_manifest.json",
    )
    parser.add_argument("--general-directory", type=Path, default=project / "data" / "training" / "general")
    parser.add_argument(
        "--dimensional-directory", type=Path, default=project / "data" / "training" / "dimensional"
    )
    parser.add_argument("--mixed-directory", type=Path, default=project / "data" / "training" / "mixed")
    parser.add_argument(
        "--extended-directory", type=Path, default=project / "data" / "training" / "extended"
    )
    parser.add_argument(
        "--large-nvt-directory", type=Path, default=project / "data" / "benchmarks" / "large_nvt"
    )
    parser.add_argument("--small-points-per-field", type=int, default=25)
    parser.add_argument("--large-points-per-field", type=int, default=100)
    parser.add_argument(
        "--temperature",
        type=float,
        choices=TEMPERATURES,
        action="append",
        help="Restrict evaluation to one or more temperatures.",
    )
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--skip-small", action="store_true")
    parser.add_argument("--skip-large", action="store_true")
    return parser.parse_args()


def empty_stats():
    return {
        "frames": 0,
        "converged": 0,
        "points": 0,
        "sse": 0.0,
        "target_sum": 0.0,
        "target_square_sum": 0.0,
    }


def add_stats(stats, target, prediction, converged=None):
    target = target.detach()
    prediction = prediction.detach()
    error = prediction - target
    stats["frames"] += 1
    if converged is not None:
        stats["converged"] += int(converged)
    stats["points"] += int(target.numel())
    stats["sse"] += float(error.square().sum())
    stats["target_sum"] += float(target.sum())
    stats["target_square_sum"] += float(target.square().sum())


def summarize(stats):
    count = stats["points"]
    rmse = math.sqrt(stats["sse"] / count)
    centered = stats["target_square_sum"] - stats["target_sum"] ** 2 / count
    r2 = math.nan if centered <= 0.0 else 1.0 - stats["sse"] / centered
    return {
        "frames": stats["frames"],
        "converged": stats["converged"],
        "points": count,
        "rmse": rmse,
        "r2": r2,
    }


def sample_pairs(target, prediction, count, rng):
    target = target.detach().cpu().numpy().reshape(-1)
    prediction = prediction.detach().cpu().numpy().reshape(-1)
    if target.size > count:
        indices = rng.choice(target.size, size=count, replace=False)
        target = target[indices]
        prediction = prediction[indices]
    return target, prediction


def load_small_test_set(GridData, arguments, grid_info, temperatures):
    manifest = json.loads(arguments.manifest.expanduser().resolve().read_text())
    roots = {
        "general": arguments.general_directory.expanduser().resolve(),
        "dimensional": arguments.dimensional_directory.expanduser().resolve(),
        "mixed": arguments.mixed_directory.expanduser().resolve(),
        "extended": arguments.extended_directory.expanduser().resolve(),
    }
    paths = []
    metadata = []
    for record in manifest["sources"]:
        if record["partition"] != "test":
            continue
        if float(record["temperature"]) not in temperatures:
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


def gauge_aligned_external_field(frame, result, model):
    rho = result["rho"].detach()
    target = frame["beta"].to(rho) * frame["V_ext"].to(rho)
    wavelength = model.thermal_wavelength.to(rho)
    prediction = result["c1"].detach() - torch.log(
        torch.clamp(rho * wavelength.pow(3), min=torch.finfo(rho.dtype).tiny)
    )
    mask = rho > model.rho_min
    denominator = mask.sum(dim=0).clamp_min(1)
    offset = ((target - prediction) * mask).sum(dim=0) / denominator
    return target[mask], (prediction + offset)[mask]


def solve_density(solver, model, frame):
    solve_data = dict(frame)
    target = solve_data.pop("rho")
    particle_numbers = torch.prod(solve_data["grid_spacing"]) * torch.sum(
        target, dim=0
    )
    solved = solver.solve(
        solve_data,
        particle_numbers=particle_numbers,
        method="euler",
        beta_multiplier=0.0,
        max_iter=200,
        tolerance_residual=1.0e-2,
        tolerance_change=1.0e-6,
        residual_density_threshold=model.rho_min,
        maximum_density=2.0,
    )
    return target, solved["rho"].detach(), bool(solved["converged"])


def output_tag(temperatures):
    if tuple(temperatures) == TEMPERATURES:
        return ""
    return "_" + "_".join(
        "T" + str(temperature).replace(".", "p") for temperature in temperatures
    )


def evaluate_small(GridData, solver, model, device, arguments, temperatures):
    frames, metadata = load_small_test_set(
        GridData, arguments, model.grid_info, temperatures
    )
    rng = np.random.default_rng(RANDOM_SEED)
    forward_stats = defaultdict(empty_stats)
    solve_stats = defaultdict(empty_stats)
    arrays = defaultdict(list)

    for index, (frame, record) in enumerate(zip(frames, metadata), start=1):
        temperature = record["temperature"]
        moved = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in frame.items()
        }
        result = solver.evaluate(moved, compute_c1=True)
        target_field, predicted_field = gauge_aligned_external_field(
            moved, result, model
        )
        add_stats(forward_stats[temperature], target_field, predicted_field)
        sampled_target, sampled_prediction = sample_pairs(
            target_field,
            predicted_field,
            arguments.small_points_per_field,
            rng,
        )
        arrays[("forward_target", temperature)].append(sampled_target)
        arrays[("forward_prediction", temperature)].append(sampled_prediction)

        target_density, predicted_density, converged = solve_density(
            solver, model, moved
        )
        add_stats(
            solve_stats[temperature],
            target_density,
            predicted_density,
            converged,
        )
        sampled_target, sampled_prediction = sample_pairs(
            target_density,
            predicted_density,
            arguments.small_points_per_field,
            rng,
        )
        arrays[("solve_target", temperature)].append(sampled_target)
        arrays[("solve_prediction", temperature)].append(sampled_prediction)

        if index % 20 == 0 or index == len(frames):
            print(
                "small NVT: {}/{} fields; converged={}/{}".format(
                    index,
                    len(frames),
                    sum(stats["converged"] for stats in solve_stats.values()),
                    index,
                ),
                flush=True,
            )

    cache = {}
    metrics = {}
    for temperature in temperatures:
        key = str(temperature).replace(".", "p")
        for name in (
            "forward_target",
            "forward_prediction",
            "solve_target",
            "solve_prediction",
        ):
            cache[f"T{key}_{name}"] = np.concatenate(arrays[(name, temperature)])
        metrics[f"T={temperature:.1f}"] = {
            "forward": summarize(forward_stats[temperature]),
            "solve": summarize(solve_stats[temperature]),
        }
    tag = output_tag(temperatures)
    np.savez_compressed(
        arguments.output_directory / ("complete_small_nvt" + tag + "_points.npz"),
        **cache,
    )
    (arguments.output_directory / ("complete_small_nvt" + tag + "_metrics.json")).write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


def evaluate_large(GridData, solver, model, device, arguments, temperatures):
    rng = np.random.default_rng(RANDOM_SEED + 1)
    stats = defaultdict(empty_stats)
    arrays = defaultdict(list)
    total = 0
    for temperature in temperatures:
        candidates = (
            arguments.large_nvt_directory / f"nvt-T{temperature:g}-0.5sigma.extxyz",
            arguments.large_nvt_directory / f"nvt-T{temperature:.1f}-0.5sigma.extxyz",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(candidates[0])
        frames = GridData.from_xyz(path, grid_info=model.grid_info)
        for frame in frames:
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in frame.items()
            }
            target, prediction, converged = solve_density(solver, model, moved)
            add_stats(stats[temperature], target, prediction, converged)
            sampled_target, sampled_prediction = sample_pairs(
                target,
                prediction,
                arguments.large_points_per_field,
                rng,
            )
            arrays[("target", temperature)].append(sampled_target)
            arrays[("prediction", temperature)].append(sampled_prediction)
            total += 1
            if total % 10 == 0:
                print("large NVT: {} fields".format(total), flush=True)

    cache = {}
    metrics = {}
    for temperature in temperatures:
        key = str(temperature).replace(".", "p")
        cache[f"T{key}_target"] = np.concatenate(arrays[("target", temperature)])
        cache[f"T{key}_prediction"] = np.concatenate(
            arrays[("prediction", temperature)]
        )
        metrics[f"T={temperature:.1f}"] = summarize(stats[temperature])
    tag = output_tag(temperatures)
    np.savez_compressed(
        arguments.output_directory / ("complete_large_nvt" + tag + "_points.npz"),
        **cache,
    )
    (arguments.output_directory / ("complete_large_nvt" + tag + "_metrics.json")).write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


def main():
    arguments = parse_arguments()
    arguments.fit_directory = arguments.fit_directory.expanduser().resolve()
    arguments.output_directory = arguments.output_directory.expanduser().resolve()
    arguments.large_nvt_directory = arguments.large_nvt_directory.expanduser().resolve()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    temperatures = tuple(arguments.temperature or TEMPERATURES)
    torch.set_num_threads(arguments.torch_threads)
    from equicdft import GridData, GridSolver

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(
        arguments.fit_directory / "model.pt",
        map_location=device,
        weights_only=False,
    ).eval()
    solver = GridSolver(model, device=device)
    if not arguments.skip_small:
        evaluate_small(GridData, solver, model, device, arguments, temperatures)
    if not arguments.skip_large:
        evaluate_large(GridData, solver, model, device, arguments, temperatures)


if __name__ == "__main__":
    main()
