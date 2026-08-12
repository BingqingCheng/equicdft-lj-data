"""Shared homogeneous-fluid response utilities for the command-line scripts."""

from pathlib import Path

import numpy as np
import torch


from equicdft import GridData  # noqa: E402


# The homogeneous k=0 EOS is numerically unchanged when the response box is
# enlarged to four times the model cutoff.  Keep the smaller, faster 10^3 box
# as the canonical default for routine compressibility-route calculations.
DEFAULT_EOS_GRID_SIZE = 10


def resolve_model_path(path):
    """Resolve either a serialized model or a directory containing model.pt."""

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_model(path, device):
    """Load a complete serialized model on the requested torch device."""

    path = resolve_model_path(path)
    model = torch.load(path, map_location=device, weights_only=False)
    return path, model.to(device).eval()


def grid_geometry(model, grid_size):
    """Return integer grid dimensions and physical spacing after validation."""

    if np.isscalar(grid_size):
        grid_size = np.full(3, int(grid_size), dtype=int)
    else:
        grid_size = np.asarray(grid_size, dtype=int)
    if grid_size.shape != (3,):
        raise ValueError("grid_size must be one integer or three integers")
    spacing = np.asarray(model.grid_info["grid_spacing"], dtype=float)
    if np.any(grid_size <= 2 * model.cutoff_grid):
        raise ValueError("each grid dimension must exceed twice cutoff_grid")
    return grid_size, spacing


def homogeneous_data(model, temperature, density, grid_size, spacing, device):
    """Construct a model input for a uniform one-component density field."""

    data = GridData.from_dict(
        {
            "grid_size": grid_size,
            "grid_spacing": spacing,
            "n_types": model.n_types,
            "temperature": temperature,
        },
        grid_info=model.grid_info,
    )
    data["rho"] = torch.full(
        (int(np.prod(grid_size)), model.n_types),
        density,
        dtype=model.grid_spacing.dtype,
    )
    return {key: value.to(device) for key, value in data.items()}


def c2_real_space(model, temperature, density, grid_size, spacing, device):
    """Return one continuum-normalized row c2(r_0,r) of the homogeneous Hessian."""

    data = homogeneous_data(
        model,
        temperature,
        density,
        grid_size,
        spacing,
        device,
    )
    with torch.no_grad():
        output = model(data, compute_c2=True, c2_reference=(0, 0))
    return output["c2"][:, 0].reshape(tuple(grid_size))


def structure_factor_grid(model, temperature, density, grid_size, device):
    """Evaluate c2(k), the OZ denominator, and S(k) on the reciprocal grid."""

    grid_size, spacing = grid_geometry(model, grid_size)
    c2_r = c2_real_space(
        model,
        temperature,
        density,
        grid_size,
        spacing,
        device,
    )
    c2_k_complex = float(np.prod(spacing)) * torch.fft.fftn(c2_r)
    c2_k = c2_k_complex.real
    denominator = 1.0 - density * c2_k
    structure_factor = 1.0 / denominator

    frequencies = [
        2.0 * np.pi * np.fft.fftfreq(n, d=h)
        for n, h in zip(grid_size, spacing)
    ]
    components = np.meshgrid(*frequencies, indexing="ij")
    k_magnitude = np.sqrt(sum(component**2 for component in components))
    return {
        "grid_size": grid_size,
        "spacing": spacing,
        "k_magnitude": k_magnitude,
        "c2_k": c2_k.detach().cpu().numpy(),
        "structure_factor": structure_factor.detach().cpu().numpy(),
        "max_imaginary_c2_k": float(c2_k_complex.imag.abs().max().cpu()),
        "minimum_oz_denominator": float(denominator.min().cpu()),
    }


def shell_average(k_magnitude, arrays, k_max):
    """Average named reciprocal-grid arrays over exactly equal |k| shells."""

    k = np.asarray(k_magnitude).ravel()
    flattened = {name: np.asarray(values).ravel() for name, values in arrays.items()}
    selected = (k > 0.0) & (k <= k_max)
    shell_key = np.round(k[selected], decimals=10)
    rows = []
    for key in np.unique(shell_key):
        members = shell_key == key
        row = {"k": k[selected][members].mean(), "n_vectors": int(members.sum())}
        for name, values in flattened.items():
            row[name] = values[selected][members].mean()
            row[name + "_std"] = values[selected][members].std()
        rows.append(row)
    return rows


def radial_structure_factor(model, temperature, density, grid_size, device, k_max=None):
    """Return radial S(k) rows and numerical diagnostics for one bulk state."""

    result = structure_factor_grid(
        model,
        temperature,
        density,
        grid_size,
        device,
    )
    if k_max is None:
        k_max = float(np.min(np.pi / result["spacing"]))
    rows = shell_average(
        result["k_magnitude"],
        {
            "S_k": result["structure_factor"],
            "c2_k": result["c2_k"],
        },
        k_max,
    )
    return rows, result


def structure_factor_zero(model, temperature, density, grid_size, device):
    """Return exact S(0) and c2(k=0) without radial extrapolation."""

    grid_size, spacing = grid_geometry(model, grid_size)
    c2_r = c2_real_space(
        model,
        temperature,
        density,
        grid_size,
        spacing,
        device,
    )
    c2_zero = float(np.prod(spacing) * c2_r.sum().cpu())
    inverse_s_zero = 1.0 - density * c2_zero
    return 1.0 / inverse_s_zero, c2_zero
