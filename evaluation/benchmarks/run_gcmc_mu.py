#!/usr/bin/env python3
"""Evaluate one NVT-trained model on one or more GCMC datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


from equicdft import GridData  # noqa: E402


def select_indices(count, fraction):
    """Return an evenly spaced deterministic fraction of one source file."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    selected_count = max(1, int(round(fraction * count)))
    if selected_count == 1:
        return [count // 2]
    return sorted(
        {
            int(round(index * (count - 1) / (selected_count - 1)))
            for index in range(selected_count)
        }
    )


def source_files(directory):
    paths = sorted(directory.expanduser().resolve().glob("*.extxyz"))
    if not paths:
        raise FileNotFoundError("no top-level EXTXYZ files in {}".format(directory))
    return paths


def evaluate_sources(model, device, collections, fraction):
    """Return one raw chemical-potential prediction per selected frame."""

    rows = []
    for collection, directory in collections:
        for path in source_files(directory):
            frames = GridData.from_xyz(path, grid_info=model.grid_info)
            for frame_index in select_indices(len(frames), fraction):
                frame = frames[frame_index]
                moved = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in frame.items()
                }
                outputs = model(moved, compute_c1=True)
                beta = float(moved["beta"].detach().cpu())
                reference_mu = float(moved["mu"].reshape(-1)[0].detach().cpu())
                predicted_beta_mu = float(
                    outputs["average_chemical_potential"]
                    .reshape(-1)[0]
                    .detach()
                    .cpu()
                )
                local_mu = outputs["local_chemical_potential"].detach()
                average_mu = outputs["average_chemical_potential"].detach()
                weights = outputs["chemical_potential_weights"].detach()
                weight_sum = float(weights.sum().cpu())
                constancy_sse = float(
                    (
                        weights
                        * (local_mu - average_mu[..., None, :]).square()
                    )
                    .sum()
                    .cpu()
                )
                rows.append(
                    {
                        "model": model_name(model),
                        "collection": collection,
                        "source_file": path.name,
                        "source_frame": frame_index,
                        "temperature": float(
                            moved["temperature"].detach().cpu()
                        ),
                        "mu_reference": reference_mu,
                        "beta_mu_reference": beta * reference_mu,
                        "mu_prediction_raw": predicted_beta_mu / beta,
                        "beta_mu_prediction_raw": predicted_beta_mu,
                        "constancy_weight": weight_sum,
                        "local_beta_mu_constancy_sse": constancy_sse,
                    }
                )
                del outputs, moved
    return rows


def model_name(model):
    return getattr(model, "_evaluation_name", model.__class__.__name__)


def align_temperature_gauges(rows):
    """Align one additive chemical-potential gauge independently at each T."""

    by_temperature = defaultdict(list)
    for row in rows:
        by_temperature[row["temperature"]].append(row)
    for temperature_rows in by_temperature.values():
        gauge_offset = float(
            np.mean(
                [
                    row["mu_reference"] - row["mu_prediction_raw"]
                    for row in temperature_rows
                ]
            )
        )
        reference_mean = float(
            np.mean([row["mu_reference"] for row in temperature_rows])
        )
        for row in temperature_rows:
            aligned = row["mu_prediction_raw"] + gauge_offset
            row["mu_gauge_offset"] = gauge_offset
            row["mu_prediction_aligned"] = aligned
            row["mu_error_aligned"] = aligned - row["mu_reference"]
            row["mu_reference_centered_at_T"] = (
                row["mu_reference"] - reference_mean
            )


def summarize(model, group, rows):
    error = np.asarray([row["mu_error_aligned"] for row in rows], dtype=float)
    target_centered = np.asarray(
        [row["mu_reference_centered_at_T"] for row in rows], dtype=float
    )
    reference = np.asarray([row["mu_reference"] for row in rows], dtype=float)
    prediction = np.asarray(
        [row["mu_prediction_aligned"] for row in rows], dtype=float
    )
    denominator = float(np.sum(target_centered**2))
    reference_std = float(np.std(reference))
    prediction_std = float(np.std(prediction))
    pearson = (
        float("nan")
        if reference_std == 0.0 or prediction_std == 0.0
        else float(np.corrcoef(reference, prediction)[0, 1])
    )
    weight = sum(row["constancy_weight"] for row in rows)
    constancy_sse = sum(
        row["local_beta_mu_constancy_sse"] for row in rows
    )
    return {
        "model": model,
        "group": group,
        "frames": len(rows),
        "temperatures": len({row["temperature"] for row in rows}),
        "unique_mu": len({row["mu_reference"] for row in rows}),
        "mu_mae_gauge_aligned": float(np.mean(np.abs(error))),
        "mu_rmse_gauge_aligned": float(np.sqrt(np.mean(error**2))),
        "mu_r2_within_temperature": (
            float("nan")
            if denominator <= 0.0
            else 1.0 - float(np.sum(error**2)) / denominator
        ),
        "mu_pearson_r": pearson,
        "local_beta_mu_constancy_rmse": math.sqrt(constancy_sse / weight),
    }


def metric_rows(model, predictions):
    rows = [summarize(model, "all", predictions)]
    temperatures = sorted({row["temperature"] for row in predictions})
    for temperature in temperatures:
        selected = [
            row for row in predictions if row["temperature"] == temperature
        ]
        rows.append(summarize(model, "T={:.1f}".format(temperature), selected))
    for collection in sorted({row["collection"] for row in predictions}):
        selected = [row for row in predictions if row["collection"] == collection]
        rows.append(summarize(model, collection, selected))
        for temperature in temperatures:
            selected_at_temperature = [
                row
                for row in selected
                if row["temperature"] == temperature
            ]
            rows.append(
                summarize(
                    model,
                    "T={:.1f}:{}".format(temperature, collection),
                    selected_at_temperature,
                )
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-directory", type=Path, default=repository / "model")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="dataset label and directory; may be supplied repeatedly",
    )
    parser.add_argument(
        "--general-directory",
        type=Path,
        default=repository / "data" / "benchmarks" / "gcmc" / "general",
    )
    parser.add_argument(
        "--dimensional-directory",
        type=Path,
        default=repository / "data" / "benchmarks" / "gcmc" / "dimensional",
    )
    parser.add_argument(
        "--mixed-directory",
        type=Path,
        default=repository / "data" / "benchmarks" / "gcmc" / "mixed",
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository / "output" / "benchmarks",
    )
    parser.add_argument("--output-prefix", default="gcmc_mu")
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_directory = arguments.fit_directory.expanduser().resolve()
    model = torch.load(
        fit_directory / "model.pt",
        map_location=device,
        weights_only=False,
    ).eval()
    model.to(device)
    model._evaluation_name = fit_directory.name
    if arguments.dataset:
        collections = []
        for specification in arguments.dataset:
            if "=" not in specification:
                parser.error("--dataset must have the form LABEL=PATH")
            label, path = specification.split("=", 1)
            collections.append((label, Path(path)))
    else:
        legacy = (
            ("general_3D", arguments.general_directory),
            ("1D2D", arguments.dimensional_directory),
            ("mixed", arguments.mixed_directory),
        )
        collections = list(legacy)
    predictions = evaluate_sources(
        model=model,
        device=device,
        collections=collections,
        fraction=arguments.fraction,
    )
    align_temperature_gauges(predictions)
    metrics = metric_rows(fit_directory.name, predictions)
    prefix = arguments.output_prefix
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(output_directory / (prefix + "_predictions.csv"), predictions)
    (output_directory / (prefix + "_metrics.json")).write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n"
    )
    overall = metrics[0]
    print(
        "{}: frames={} gauge-aligned mu RMSE={:.6f} R2={:.6f}".format(
            fit_directory.name,
            overall["frames"],
            overall["mu_rmse_gauge_aligned"],
            overall["mu_r2_within_temperature"],
        )
    )


if __name__ == "__main__":
    main()
