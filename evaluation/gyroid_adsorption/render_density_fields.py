#!/usr/bin/env python3
"""Render paper-ready 3D and slice comparisons of MD and cDFT densities."""

from pathlib import Path
import argparse
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


# Set from the command-line results directory in ``main``. Keeping this as a
# module variable also lets the specialized gyroid renderer reuse the common
# loading and marching-cubes helpers.
DIRECTORY = Path.cwd()
SOURCE_DATA = Path(__file__).resolve().parents[2] / "source_data" / "fig04_gyroid_adsorption" / "density_fields.npz"
SYSTEMS = (("gyroid-pore", "Gyroid pore"),)
GRID_SPACING = 0.5
EXTERNAL_LEVEL = 4.0
RENDER_SMOOTHING = 0.45

# Application-specific levels expose structure that a single global scale hides.
# Density uses distinct palettes for the three geometries.  The error palettes
# retain the same sign convention throughout: warm is positive and cool is
# negative.  Increasing saturation denotes increasing contour magnitude.
VISUAL_STYLES = {
    "wavy-wall": {
        "density_levels": (0.04, 0.25, 0.65),
        "density_colors": ("#73D2FF", "#0077B6", "#002855"),
        "error_levels": (0.008, 0.018, 0.032),
    },
    "gyroid-pore": {
        "density_levels": (0.05, 0.30, 0.66),
        "density_colors": ("#A7F3A1", "#00A878", "#004B3A"),
        "error_levels": (0.012, 0.028, 0.045),
    },
    "chemically-patterned-wall": {
        "density_levels": (0.04, 0.28, 0.65),
        "density_colors": ("#F0A6FF", "#B5179E", "#48006A"),
        "error_levels": (0.008, 0.018, 0.032),
    },
}
DENSITY_ALPHAS = (0.16, 0.43, 0.96)
ERROR_POSITIVE_COLORS = ("#FFD166", "#F77F00", "#C1121F")
ERROR_NEGATIVE_COLORS = ("#67E8F9", "#168AAD", "#253494")
ERROR_ALPHAS = (0.20, 0.52, 0.96)


def reshape_grid(atoms, values):
    grid_size = tuple(np.asarray(atoms.info["grid_size"], dtype=int))
    positions = np.rint(atoms.positions).astype(int)
    if str(atoms.info.get("grid_indexing", "zero_based")).lower().startswith("one"):
        positions -= 1
    grid = np.empty(grid_size, dtype=float)
    grid[positions[:, 0], positions[:, 1], positions[:, 2]] = values
    return grid


def load_fields():
    with np.load(SOURCE_DATA) as data:
        reference = data["reference"]
        prediction = data["prediction"]
        external = data["external"]
    error = prediction - reference
    rmse = float(np.sqrt(np.mean(error**2)))
    return [{
        "name": "gyroid-pore",
        "label": "Gyroid pore",
        "reference": reference,
        "prediction": prediction,
        "error": error,
        "external": external,
        "rmse": rmse,
        "r2": 1.0 - rmse**2 / float(np.var(reference)),
        "lengths": np.asarray(reference.shape) * GRID_SPACING,
    }]


def periodic_surface(field, level):
    smoothed = gaussian_filter(field, RENDER_SMOOTHING, mode="wrap")
    periodic = np.pad(smoothed, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    if not float(np.min(periodic)) < level < float(np.max(periodic)):
        return None, None
    vertices, faces, _, _ = marching_cubes(
        periodic,
        level=level,
        spacing=(GRID_SPACING,) * 3,
        allow_degenerate=False,
    )
    return vertices, faces


def shaded_colors(triangles, color, alpha):
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edges_a, edges_b)
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norms[:, None], np.finfo(float).eps)
    light = np.asarray((0.6, -0.8, 1.25), dtype=float)
    light /= np.linalg.norm(light)
    illumination = 0.82 + 0.18 * np.clip(normals @ light, -0.25, 1.0)
    rgb = np.asarray(to_rgb(color))[None, :] * illumination[:, None]
    rgba = np.empty((len(triangles), 4), dtype=float)
    rgba[:, :3] = np.clip(rgb, 0.0, 1.0)
    rgba[:, 3] = alpha
    return rgba


def add_surface(axis, field, level, color, alpha):
    vertices, faces = periodic_surface(field, level)
    if vertices is None:
        return
    triangles = vertices[faces]
    collection = Poly3DCollection(
        triangles,
        facecolors=shaded_colors(triangles, color, alpha),
        edgecolors="none",
        linewidths=0.0,
        zsort="average",
    )
    axis.add_collection3d(collection)


def draw_box(axis, lengths):
    corners = np.asarray(
        [
            (x, y, z)
            for x in (0.0, lengths[0])
            for y in (0.0, lengths[1])
            for z in (0.0, lengths[2])
        ]
    )
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(corners[first] != corners[second]) == 1:
                points = corners[[first, second]]
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    color="0.35",
                    linewidth=0.45,
                    alpha=0.55,
                )


def style_3d_axis(axis, lengths):
    axis.set_xlim(0.0, lengths[0])
    axis.set_ylim(0.0, lengths[1])
    axis.set_zlim(0.0, lengths[2])
    axis.set_box_aspect(lengths, zoom=1.22)
    axis.set_proj_type("ortho")
    axis.view_init(elev=24, azim=-55)
    axis.set_axis_off()
    draw_box(axis, lengths)


def draw_3d_panel(axis, field, quantity):
    style = VISUAL_STYLES[field["name"]]
    add_surface(axis, field["external"], EXTERNAL_LEVEL, "#7A7A7A", 0.075)
    if quantity in ("reference", "prediction"):
        for level, color, alpha in zip(
            style["density_levels"],
            style["density_colors"],
            DENSITY_ALPHAS,
        ):
            add_surface(axis, field[quantity], level, color, alpha)
    else:
        for level, positive, negative, alpha in zip(
            style["error_levels"],
            ERROR_POSITIVE_COLORS,
            ERROR_NEGATIVE_COLORS,
            ERROR_ALPHAS,
        ):
            add_surface(axis, field["error"], level, positive, alpha)
            add_surface(axis, field["error"], -level, negative, alpha)
    style_3d_axis(axis, field["lengths"])


def render_3d_panel_file(field, quantity, path, pixels=850):
    """Render one 3D panel; callers isolate each panel in a fresh process."""

    figure = plt.figure(figsize=(4.0, 4.0))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    draw_3d_panel(axis, field, quantity)
    figure.savefig(path, dpi=pixels / 4.0)
    plt.close(figure)


def render_3d_images(fields):
    """Render every panel in a separate process and return image arrays."""

    quantities = ("reference", "prediction", "error")
    images = {}
    with tempfile.TemporaryDirectory(prefix="equicdft-density-render-") as temporary:
        temporary = Path(temporary)
        for field in fields:
            for quantity in quantities:
                path = temporary / (field["name"] + "-" + quantity + ".png")
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        str(DIRECTORY),
                        "--render-panel",
                        field["name"],
                        quantity,
                        str(path),
                    ],
                    check=True,
                )
                images[(field["name"], quantity)] = plt.imread(path)
    return images


def three_dimensional_figure(fields, images):
    quantities = ("reference", "prediction", "error")
    figure, axes = plt.subplots(3, 3, figsize=(10.8, 10.2))
    for row, field in enumerate(fields):
        for column, quantity in enumerate(quantities):
            axes[row, column].imshow(images[(field["name"], quantity)])
            axes[row, column].set_axis_off()
        axes[row, 1].text(
            0.02,
            0.03,
            "RMSE {:.4f}\n$R^2$ {:.4f}".format(field["rmse"], field["r2"]),
            transform=axes[row, 1].transAxes,
            fontsize=8.5,
        )

    for x, title in zip((0.205, 0.50, 0.795), ("MD", "cDFT", r"$\Delta\rho$")):
        figure.text(x, 0.945, title, ha="center", va="center", fontsize=13)
    for y, field in zip((0.795, 0.505, 0.215), fields):
        figure.text(0.035, y, field["label"], rotation=90, ha="center", va="center", fontsize=12)

    density_handles = [
        Patch(color="#7A7A7A", alpha=0.30, label=r"$V_{\rm ext}=4$"),
        Patch(color=ERROR_POSITIVE_COLORS[-1], label=r"$\Delta\rho>0$"),
        Patch(color=ERROR_NEGATIVE_COLORS[-1], label=r"$\Delta\rho<0$"),
    ]
    figure.legend(
        handles=density_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.52, 0.025),
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.91, wspace=0.0, hspace=0.0)
    figure.savefig(DIRECTORY / "density_fields_3d_comparison.png", dpi=350)
    figure.savefig(DIRECTORY / "density_fields_3d_comparison.pdf")
    plt.close(figure)

    for field in fields:
        figure, axes = plt.subplots(1, 3, figsize=(11.6, 4.0))
        for axis, quantity, title in zip(axes, quantities, ("MD", "cDFT", r"$\Delta\rho$")):
            axis.imshow(images[(field["name"], quantity)])
            axis.set_axis_off()
            axis.set_title(title, pad=3)
        axes[1].text(
            0.02,
            0.02,
            "RMSE {:.4f}\n$R^2$ {:.4f}".format(field["rmse"], field["r2"]),
            transform=axes[1].transAxes,
            fontsize=9,
        )
        style = VISUAL_STYLES[field["name"]]
        figure.text(
            0.5,
            0.105,
            r"$\rho$: {} (light$\rightarrow$dark)    $|\Delta\rho|$: {} (light$\rightarrow$dark)".format(
                ", ".join("{:.2f}".format(value) for value in style["density_levels"]),
                ", ".join("{:.3f}".format(value) for value in style["error_levels"]),
            ),
            ha="center",
            fontsize=8.5,
        )
        figure.suptitle(field["label"], y=0.98, fontsize=13)
        figure.legend(
            handles=density_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.17, top=0.90, wspace=0.0)
        stem = DIRECTORY / ("density_fields_3d_" + field["name"])
        figure.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def best_slice(field):
    best = None
    for axis in range(3):
        for index in range(field["reference"].shape[axis]):
            plane = np.take(field["reference"], index, axis=axis)
            score = float(np.var(plane))
            if best is None or score > best[0]:
                best = (score, axis, index)
    return best[1], best[2]


def slice_axes(axis):
    labels = (("y", "z"), ("x", "z"), ("x", "y"))
    return labels[axis]


def slice_figure(fields):
    figure, axes = plt.subplots(3, 3, figsize=(10.6, 9.6))
    density_image = None
    error_image = None
    for row, field in enumerate(fields):
        slice_axis, slice_index = best_slice(field)
        length_axes = [index for index in range(3) if index != slice_axis]
        extent = (
            0.0,
            field["lengths"][length_axes[0]],
            0.0,
            field["lengths"][length_axes[1]],
        )
        external_plane = np.take(field["external"], slice_index, axis=slice_axis).T
        for column, quantity in enumerate(("reference", "prediction", "error")):
            plane = np.take(field[quantity], slice_index, axis=slice_axis).T
            if quantity == "error":
                error_image = axes[row, column].imshow(
                    plane,
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-0.05,
                    vmax=0.05,
                    interpolation="bilinear",
                )
            else:
                density_image = axes[row, column].imshow(
                    plane,
                    origin="lower",
                    extent=extent,
                    cmap="viridis",
                    vmin=0.0,
                    vmax=0.80,
                    interpolation="bilinear",
                )
            axes[row, column].contour(
                external_plane,
                levels=(EXTERNAL_LEVEL,),
                origin="lower",
                extent=extent,
                colors="white" if quantity != "error" else "black",
                linewidths=0.65,
                alpha=0.75,
            )
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlim(extent[:2])
            axes[row, column].set_ylim(extent[2:])
        first_label, second_label = slice_axes(slice_axis)
        axes[row, 0].set_ylabel(r"${}\,/\,\sigma$".format(second_label))
        axes[row, 0].text(
            -0.29,
            0.5,
            "{}\n{} = {:.2f}$\\sigma$".format(
                field["label"],
                "xyz"[slice_axis],
                slice_index * GRID_SPACING,
            ),
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
        )
        for column in range(3):
            axes[-1, column].set_xlabel(r"${}\,/\,\sigma$".format(first_label))

    for axis, title in zip(axes[0], ("MD", "cDFT", r"$\Delta\rho$")):
        axis.set_title(title)
    density_bar_axis = figure.add_axes((0.16, 0.055, 0.48, 0.022))
    density_bar = figure.colorbar(density_image, cax=density_bar_axis, orientation="horizontal")
    density_bar.set_label(r"Density, $\rho\sigma^3$")
    error_bar_axis = figure.add_axes((0.72, 0.055, 0.22, 0.022))
    error_bar = figure.colorbar(error_image, cax=error_bar_axis, orientation="horizontal")
    error_bar.set_label(r"$\Delta\rho\sigma^3$")
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.13, top=0.94, wspace=0.12, hspace=0.18)
    figure.savefig(DIRECTORY / "density_fields_slice_comparison.png", dpi=350)
    figure.savefig(DIRECTORY / "density_fields_slice_comparison.pdf")
    plt.close(figure)


def main():
    global DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_directory",
        type=Path,
        help="model applications-soft directory containing *-solved.extxyz",
    )
    parser.add_argument(
        "--render-panel",
        nargs=3,
        metavar=("SYSTEM", "QUANTITY", "OUTPUT"),
    )
    arguments = parser.parse_args()
    DIRECTORY = arguments.results_directory.expanduser().resolve()
    fields = load_fields()
    if arguments.render_panel is not None:
        name, quantity, output = arguments.render_panel
        field = next(field for field in fields if field["name"] == name)
        render_3d_panel_file(field, quantity, Path(output))
        return
    images = render_3d_images(fields)
    three_dimensional_figure(fields, images)
    slice_figure(fields)


if __name__ == "__main__":
    main()
