#!/usr/bin/env python3
"""Integrate the homogeneous compressibility route to obtain an LJ isotherm."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import DEFAULT_EOS_GRID_SIZE, load_model, structure_factor_zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="model.pt or its directory")
    parser.add_argument("reference_eos", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--rho-max", type=float, default=1.1)
    density_mesh = parser.add_mutually_exclusive_group()
    density_mesh.add_argument("--n-rho", type=int)
    density_mesh.add_argument("--rho-step", type=float, default=0.005)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_EOS_GRID_SIZE,
        help=(
            "number of grid points along each axis "
            f"(default: {DEFAULT_EOS_GRID_SIZE})"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    _model_path, model = load_model(args.model, device)

    grid_size = np.full(3, args.grid_size, dtype=int)

    if args.n_rho is None:
        n_rho = int(round(args.rho_max / args.rho_step)) + 1
    else:
        n_rho = args.n_rho
    density = np.linspace(0.0, args.rho_max, n_rho)
    s_zero = np.empty_like(density)
    c2_zero = np.empty_like(density)
    for index, rho in enumerate(density):
        s_zero[index], c2_zero[index] = structure_factor_zero(
            model,
            args.temperature,
            float(rho),
            grid_size,
            device,
        )

    inverse_s_zero = 1.0 / s_zero
    beta_pressure = np.zeros_like(density)
    beta_pressure[1:] = np.cumsum(
        0.5
        * (inverse_s_zero[1:] + inverse_s_zero[:-1])
        * np.diff(density)
    )
    pressure = args.temperature * beta_pressure

    reference = np.genfromtxt(
        args.reference_eos.expanduser().resolve(),
        delimiter=",",
        names=True,
    )
    selected = np.isclose(reference["target_temperature"], args.temperature)
    reference = reference[selected]
    reference = reference[reference["density"] <= args.rho_max]
    if len(reference):
        predicted_reference = np.interp(reference["density"], density, pressure)
        error = predicted_reference - reference["pressure"]
    else:
        predicted_reference = np.empty(0)
        error = np.empty(0)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        Path(str(output) + ".csv"),
        np.column_stack(
            [density, c2_zero, s_zero, inverse_s_zero, beta_pressure, pressure]
        ),
        delimiter=",",
        header="density,c2_k0,S_k0,inverse_S_k0,beta_pressure,pressure",
        comments="",
        fmt="%.10g",
    )
    np.savetxt(
        Path(str(output) + "_reference_comparison.csv"),
        np.column_stack(
            [
                reference["density"],
                reference["pressure"],
                reference["pressure_standard_error"],
                predicted_reference,
                error,
            ]
        ),
        delimiter=",",
        header="density,reference_pressure,reference_standard_error,model_pressure,error",
        comments="",
        fmt="%.10g",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].plot(density, s_zero, color="#0072B2", linewidth=1.7)
    axes[0].axhline(1.0, color="0.45", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel(r"number density $\rho\sigma^3$")
    axes[0].set_ylabel(r"$S(0)$")
    axes[0].set_title("Homogeneous response")
    axes[0].grid(alpha=0.20)

    if len(reference):
        axes[1].errorbar(
            reference["density"],
            reference["pressure"],
            yerr=reference["pressure_standard_error"],
            color="0.20",
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            capsize=1.5,
            label="Atomistic MD",
        )
    axes[1].plot(
        density,
        pressure,
        color="#0072B2",
        linewidth=1.8,
        label="cDFT compressibility route",
    )
    axes[1].set_xlabel(r"number density $\rho\sigma^3$")
    axes[1].set_ylabel(r"pressure $P\sigma^3/\epsilon$")
    axes[1].set_title(r"$T^*={:g}$".format(args.temperature))
    axes[1].grid(alpha=0.20)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(Path(str(output) + ".png"), dpi=260)
    fig.savefig(Path(str(output) + ".pdf"))
    plt.close(fig)

    print("reference points:", len(reference))
    if len(reference):
        print("MAE: {:.6f}".format(np.mean(np.abs(error))))
        print("RMSE: {:.6f}".format(np.sqrt(np.mean(error**2))))
    print("minimum S(0): {:.6f}".format(s_zero.min()))
    print("maximum S(0): {:.6f}".format(s_zero.max()))
    print("minimum 1/S(0): {:.6f}".format(inverse_s_zero.min()))


if __name__ == "__main__":
    main()
