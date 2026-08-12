#!/usr/bin/env python3
"""Compute radial S(k) for one homogeneous state of a learned cDFT model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import load_model, radial_structure_factor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="model.pt or its directory")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--k-max", type=float)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_path, model = load_model(args.model, device)
    rows, diagnostics = radial_structure_factor(
        model,
        args.temperature,
        args.density,
        args.grid_size,
        device,
        args.k_max,
    )
    values = np.asarray(
        [
            [row["k"], row["S_k"], row["S_k_std"], row["c2_k"], row["n_vectors"]]
            for row in rows
        ]
    )

    output = args.output
    if output is None:
        output = model_path.parent / "structure_factor_T{:g}_rho{:g}".format(
            args.temperature,
            args.density,
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        Path(str(output) + ".csv"),
        values,
        delimiter=",",
        header="k,S_k,S_k_std,c2_k,n_vectors",
        comments="",
        fmt=("%.10g", "%.10g", "%.10g", "%.10g", "%d"),
    )

    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    axis.plot(values[:, 0], values[:, 1], "o-", markersize=3, linewidth=1.2)
    axis.axhline(1.0, color="0.55", linewidth=0.8)
    axis.set_xlabel(r"$k=|\mathbf{k}|\;[\sigma^{-1}]$")
    axis.set_ylabel(r"$S(k)$")
    axis.set_title(r"$T^*={:g},\;\rho^*={:g}$".format(args.temperature, args.density))
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(str(output) + ".png"), dpi=250)
    fig.savefig(Path(str(output) + ".pdf"))
    plt.close(fig)

    box_length = diagnostics["grid_size"] * diagnostics["spacing"]
    print("model: {}".format(model_path))
    print("box: {} sigma".format(" x ".join("{:g}".format(x) for x in box_length)))
    print("max |Im c2(k)|: {:.3e}".format(diagnostics["max_imaginary_c2_k"]))
    print("minimum OZ denominator: {:.6g}".format(diagnostics["minimum_oz_denominator"]))
    print("output prefix: {}".format(output))


if __name__ == "__main__":
    main()
