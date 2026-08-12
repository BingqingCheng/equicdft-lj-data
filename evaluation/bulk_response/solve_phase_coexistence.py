#!/usr/bin/env python3
"""Map liquid-vapor coexistence from one fixed-N slab initialization."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read


ROOT = Path(__file__).resolve().parents[2]
START_SLAB = (
    ROOT
    / "data"
    / "bulk_response"
    / "slabs"
    / "nvt-slab-L6x6x24-T1.05-0.5sigma.extxyz"
)
OUTPUT_STEM = "liquid_vapor_phase_diagram_slab_seed"
MAXIMUM_PHASE_DIAGRAM_TEMPERATURE = 1.05


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def temperature_grid(minimum, maximum, step):
    """Return an inclusive temperature grid without accumulated roundoff."""

    count = int(round((maximum - minimum) / step))
    grid = minimum + step * np.arange(count + 1, dtype=float)
    if not np.isclose(grid[-1], maximum):
        raise ValueError("temperature range must be an integer number of steps")
    return grid


def planar_profile(rho, grid_size):
    """Average the three-dimensional field over the interface plane."""

    shape = tuple(int(value) for value in grid_size)
    return np.asarray(rho, dtype=float).reshape(shape).mean(axis=(0, 1))


def plateau_densities(profile, fraction=0.20):
    """Estimate vapor and liquid plateaus from the profile's outer quantiles."""

    ordered = np.sort(np.asarray(profile, dtype=float))
    count = max(2, int(round(fraction * len(ordered))))
    return float(ordered[:count].mean()), float(ordered[-count:].mean())


def scalar(value):
    """Convert a scalar tensor or Python number to ``float``."""

    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def make_solve_data(GridData, model, grid_size, temperature, device):
    data = GridData.from_dict(
        {
            "grid_size": list(grid_size),
            "temperature": float(temperature),
            "n_types": 1,
            "grid_spacing": model.grid_info["grid_spacing"],
        },
        grid_info=model.grid_info,
    )
    data = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data.items()
    }
    n_grid = int(np.prod(grid_size))
    dtype = next(model.parameters()).dtype
    data["V_ext"] = torch.zeros((n_grid, 1), dtype=dtype, device=device)
    return data


def solve_temperature(
    GridData,
    solver,
    model,
    grid_size,
    temperature,
    initial_rho,
    particle_number,
    args,
):
    data = make_solve_data(
        GridData,
        model,
        grid_size,
        temperature,
        solver.device,
    )
    result = solver.solve(
        data,
        initial_rho=initial_rho,
        particle_numbers=[particle_number],
        method="euler",
        max_iter=args.max_iter,
        tolerance_residual=args.tolerance_residual,
        tolerance_rms_residual=args.tolerance_rms_residual,
        tolerance_change=args.tolerance_change,
        mixing=0.05,
        adaptive_mixing=True,
        minimum_mixing=0.005,
        maximum_mixing=0.2,
        mixing_growth=1.1,
        mixing_backtrack_factor=0.5,
        max_log_density_change=2.0,
        residual_density_threshold=getattr(model, "rho_min", 1.0e-3),
        maximum_density=args.maximum_density,
    )

    rho = result["rho"].detach()
    profile = planar_profile(rho[:, 0].cpu().numpy(), grid_size)
    vapor, liquid = plateau_densities(profile)
    row = {
        "temperature": float(temperature),
        "rho_vapor": vapor,
        "rho_liquid": liquid,
        "density_contrast": liquid - vapor,
        "mean_density": float(profile.mean()),
        "converged": bool(result["converged"]),
        "iterations": int(result["n_iter"]),
        "max_residual": scalar(result["max_euler_lagrange_residual"]),
        "rms_residual": scalar(result["rms_euler_lagrange_residual"]),
    }
    print(
        "T={:.3f} converged={} iterations={} rho=({:.4f}, {:.4f})".format(
            temperature,
            row["converged"],
            row["iterations"],
            vapor,
            liquid,
        ),
        flush=True,
    )
    return row, rho


def reference_points(directory):
    """Read available direct-MD slab plateaus for visual comparison."""

    rows = []
    for path in sorted(directory.glob("nvt-slab-L6x6x24-T*-0.5sigma.extxyz")):
        atoms = read(path)
        grid_size = np.asarray(atoms.info["grid_size"], dtype=int)
        profile = planar_profile(atoms.arrays["density"], grid_size)
        vapor, liquid = plateau_densities(profile)
        rows.append((float(atoms.info["T"]), vapor, liquid))
    return rows


def write_results(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_critical_model(temperature, vapor, liquid, converged, minimum_gap):
    """Fit a mean-field binodal using only the solved subcritical states.

    The order parameter and rectilinear diameter are fitted as

    ``(rho_l-rho_v)^2 = a T + b`` and
    ``(rho_l+rho_v)/2 = c T + d``.

    Their extrapolated zero-gap intersection gives the predicted critical
    point.  This is an extrapolation, not a self-consistent solve above 1.05.
    """

    gap = liquid - vapor
    selected = converged & (gap >= minimum_gap) & (gap <= 0.55)
    if np.count_nonzero(selected) < 4:
        return None
    gap_slope, gap_intercept = np.polyfit(
        temperature[selected],
        gap[selected] ** 2,
        deg=1,
    )
    if gap_slope >= 0.0:
        return None
    critical_temperature = -gap_intercept / gap_slope
    diameter = 0.5 * (vapor + liquid)
    diameter_slope, diameter_intercept = np.polyfit(
        temperature[selected],
        diameter[selected],
        deg=1,
    )
    critical_density = (
        diameter_slope * critical_temperature + diameter_intercept
    )
    return {
        "critical_temperature": float(critical_temperature),
        "critical_density": float(critical_density),
        "gap_slope": float(gap_slope),
        "gap_intercept": float(gap_intercept),
        "diameter_slope": float(diameter_slope),
        "diameter_intercept": float(diameter_intercept),
        "maximum_fitted_temperature": float(np.max(temperature[selected])),
    }


def plot_results(path, model_name, rows, reference, contrast_threshold):
    temperature = np.asarray([row["temperature"] for row in rows])
    vapor = np.asarray([row["rho_vapor"] for row in rows])
    liquid = np.asarray([row["rho_liquid"] for row in rows])
    contrast = np.asarray([row["density_contrast"] for row in rows])
    converged = np.asarray([row["converged"] for row in rows], dtype=bool)
    separated = converged & (contrast >= contrast_threshold)

    fig, axis = plt.subplots(figsize=(5.6, 4.5))
    if np.any(separated):
        order = np.argsort(temperature[separated])
        axis.plot(
            vapor[separated][order],
            temperature[separated][order],
            "o-",
            color="#0072B2",
            markersize=3.5,
            linewidth=1.5,
            label="cDFT vapor",
        )
        axis.plot(
            liquid[separated][order],
            temperature[separated][order],
            "s-",
            color="#D55E00",
            markersize=3.2,
            linewidth=1.5,
            label="cDFT liquid",
        )

    critical_model = fit_critical_model(
        temperature,
        vapor,
        liquid,
        converged,
        contrast_threshold,
    )
    if critical_model is not None:
        critical_temperature = critical_model["critical_temperature"]
        critical_density = critical_model["critical_density"]
        extrapolation_temperature = np.linspace(
            critical_model["maximum_fitted_temperature"],
            critical_temperature,
            80,
        )
        gap_squared = (
            critical_model["gap_slope"] * extrapolation_temperature
            + critical_model["gap_intercept"]
        )
        gap = np.sqrt(np.maximum(gap_squared, 0.0))
        diameter = (
            critical_model["diameter_slope"] * extrapolation_temperature
            + critical_model["diameter_intercept"]
        )
        axis.plot(
            diameter - 0.5 * gap,
            extrapolation_temperature,
            "--",
            color="#0072B2",
            linewidth=1.4,
            label="mean-field extrapolation",
        )
        axis.plot(
            diameter + 0.5 * gap,
            extrapolation_temperature,
            "--",
            color="#D55E00",
            linewidth=1.4,
        )
        axis.scatter(
            critical_density,
            critical_temperature,
            marker="*",
            s=85,
            color="0.20",
            label="predicted critical point",
            zorder=4,
        )
        axis.text(
            critical_density + 0.02,
            critical_temperature,
            r"$T_c^*={:.3f}$".format(critical_temperature),
            ha="left",
            va="center",
            color="0.25",
            fontsize=8,
        )
    if reference:
        reference = np.asarray(reference, dtype=float)
        axis.scatter(
            reference[:, 1],
            reference[:, 0],
            facecolors="none",
            edgecolors="0.25",
            marker="o",
            s=28,
            label="MD slab vapor",
        )
        axis.scatter(
            reference[:, 2],
            reference[:, 0],
            facecolors="none",
            edgecolors="0.25",
            marker="s",
            s=26,
            label="MD slab liquid",
        )

    failed_count = int(np.count_nonzero(~converged))
    if failed_count:
        axis.text(
            0.02,
            0.98,
            "{} non-converged temperatures omitted".format(failed_count),
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="0.35",
            fontsize=8,
        )

    axis.set_xlabel(r"number density $\rho\sigma^3$")
    axis.set_ylabel(r"temperature $T^*$")
    axis.set_title(model_name)
    axis.set_xlim(0.0, 0.90)
    if len(temperature) > 1:
        plot_maximum = 1.20
        if critical_model is not None:
            plot_maximum = max(
                plot_maximum,
                critical_model["critical_temperature"] + 0.03,
            )
        axis.set_ylim(float(temperature.min()), plot_maximum)
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(str(path) + ".png"), dpi=260)
    fig.savefig(Path(str(path) + ".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=ROOT / "model", help="model.pt or its folder")
    parser.add_argument("--start-slab", type=Path, default=START_SLAB)
    parser.add_argument("--temperature-min", type=float, default=0.65)
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=MAXIMUM_PHASE_DIAGRAM_TEMPERATURE,
    )
    parser.add_argument("--temperature-step", type=float, default=0.025)
    parser.add_argument("--contrast-threshold", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--maximum-density", type=float, default=2.0)
    parser.add_argument("--tolerance-residual", type=float, default=3.0e-2)
    parser.add_argument("--tolerance-rms-residual", type=float, default=1.0e-2)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-5)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="omit the direct-MD slab points from the plot",
    )
    args = parser.parse_args()

    if args.temperature_max > MAXIMUM_PHASE_DIAGRAM_TEMPERATURE:
        parser.error(
            "phase-diagram calculations are restricted to T* <= {:.2f}".format(
                MAXIMUM_PHASE_DIAGRAM_TEMPERATURE
            )
        )

    from equicdft import GridData, GridSolver

    device = torch.device(args.device)
    model_path = args.model.expanduser().resolve()
    if model_path.is_dir():
        model_folder = model_path
        model_path = model_folder / "model.pt"
    else:
        model_folder = model_path.parent
    model = torch.load(model_path, map_location=device, weights_only=False).eval()
    if not hasattr(model, "compute_c2"):
        model.compute_c2 = False
    model.to(device)
    solver = GridSolver(model, device=device)

    start_slab = args.start_slab.expanduser().resolve()
    start_frame = GridData.from_xyz(start_slab, grid_info=model.grid_info)[0]
    grid_size = np.asarray(start_frame["grid_size"], dtype=int)
    start_rho = start_frame["rho"].to(
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    cell_volume = float(torch.prod(start_frame["grid_spacing"]))
    particle_number = cell_volume * float(torch.sum(start_rho))

    temperatures = temperature_grid(
        args.temperature_min,
        args.temperature_max,
        args.temperature_step,
    )
    rows = []
    for temperature in temperatures:
        row, _solution = solve_temperature(
            GridData,
            solver,
            model,
            grid_size,
            float(temperature),
            start_rho,
            particle_number,
            args,
        )
        rows.append(row)
    output = args.output
    if output is None:
        output = ROOT / "output" / "bulk_response" / OUTPUT_STEM
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_results(Path(str(output) + ".csv"), rows)
    reference = [] if args.no_reference else reference_points(start_slab.parent)
    plot_results(
        output,
        model_folder.name,
        rows,
        reference,
        args.contrast_threshold,
    )
    print("wrote {}.[csv,png,pdf]".format(output), flush=True)


if __name__ == "__main__":
    main()
