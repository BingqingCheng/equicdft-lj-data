#!/usr/bin/env python3
"""Render a high-contrast multilevel isosurface comparison for the gyroid pore."""

from pathlib import Path
import argparse
from itertools import product
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.spatial import cKDTree

import render_density_fields as density_renderer
from render_density_fields import (
    EXTERNAL_LEVEL,
    GRID_SPACING,
    add_surface,
    periodic_surface,
)


# Eight rounded density surfaces resolve the dilute pore boundary, interfacial
# region, and dense liquid. The high-contrast turbo map is sampled at these
# physical contour values and shown explicitly by the density colorbar.
DENSITY_LEVELS = np.asarray((0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70))
DENSITY_ALPHAS = np.asarray((0.09, 0.12, 0.17, 0.23, 0.31, 0.42, 0.62, 0.92))
DENSITY_CMAP = plt.get_cmap("turbo")
DENSITY_NORM = Normalize(vmin=float(DENSITY_LEVELS[0]), vmax=float(DENSITY_LEVELS[-1]))
FONT_SIZE = 11.0
CONTOUR_PANEL_SCALE = 0.85
PARITY_PANEL_SCALE = 1.10
DENSITY_FIGURE_HEIGHT_INCHES = 3.45
# Cumulative downward translation from the original production layout.
LEFT_BLOCK_VERTICAL_SHIFT_CM = -1.0
LEFT_BLOCK_VERTICAL_SHIFT = LEFT_BLOCK_VERTICAL_SHIFT_CM / (
    2.54 * DENSITY_FIGURE_HEIGHT_INCHES
)
# Raise only the contour pair within the translated block to tighten its gap
# below the density colorbar while preserving the block's aligned upper edge.
CONTOUR_GAP_ADJUSTMENT = 0.030
RESULTS_DIRECTORY = Path.cwd()

OUTPUT_STEMS = {
    "gyroid-pore": "gyroid_density_multicontour",
}
def application_field(system):
    return next(
        field for field in density_renderer.load_fields() if field["name"] == system
    )


def minimum_wall_distance(external):
    """Return periodic distance to the nominal ``V_ext = 4`` pore wall."""

    vertices, faces = periodic_surface(external, EXTERNAL_LEVEL)
    triangles = vertices[faces]
    surface_samples = np.vstack(
        (
            vertices,
            np.mean(triangles, axis=1),
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        )
    )
    lengths = np.asarray(external.shape, dtype=float) * GRID_SPACING
    shifts = np.asarray(tuple(product((-1, 0, 1), repeat=3)), dtype=float) * lengths
    periodic_samples = (surface_samples[None, :, :] + shifts[:, None, :]).reshape(-1, 3)

    grid_coordinates = (
        np.moveaxis(np.indices(external.shape, dtype=float), 0, -1).reshape(-1, 3)
        * GRID_SPACING
    )
    distances, _ = cKDTree(periodic_samples).query(grid_coordinates)
    return distances.reshape(external.shape)


def style_panel(axis, lengths):
    """Apply the common camera without drawing a box or orientation cube."""

    axis.set_xlim(0.0, lengths[0])
    axis.set_ylim(0.0, lengths[1])
    axis.set_zlim(0.0, lengths[2])
    axis.set_box_aspect(lengths, zoom=1.22)
    axis.set_proj_type("ortho")
    axis.view_init(elev=24, azim=-55)
    axis.set_axis_off()


def draw_panel(axis, field, quantity):
    """Draw the common set of density contours for one density field."""

    for level, alpha in zip(DENSITY_LEVELS, DENSITY_ALPHAS):
        add_surface(
            axis,
            field[quantity],
            float(level),
            DENSITY_CMAP(DENSITY_NORM(level)),
            float(alpha),
        )
    style_panel(axis, field["lengths"])


def render_panel(system, quantity, output, pixels=1000):
    figure = plt.figure(figsize=(4.4, 4.4))
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    draw_panel(axis, application_field(system), quantity)
    figure.savefig(
        output,
        dpi=pixels / 4.4,
    )
    plt.close(figure)


def render_images(system):
    images = {}
    with tempfile.TemporaryDirectory(prefix="equicdft-gyroid-render-") as temporary:
        temporary = Path(temporary)
        for quantity in ("reference", "prediction"):
            output = temporary / (quantity + ".png")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(RESULTS_DIRECTORY),
                    "--system",
                    system,
                    "--panel",
                    quantity,
                    str(output),
                ],
                check=True,
            )
            images[quantity] = plt.imread(output)
    return images


def assemble_figure(system, images):
    field = application_field(system)
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.50, DENSITY_FIGURE_HEIGHT_INCHES))
    outer_grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.85, 0.91),
        left=0.01,
        right=0.975,
        bottom=0.11,
        top=0.79,
        wspace=0.20,
    )
    density_grid = outer_grid[0, 0].subgridspec(1, 2, wspace=0.0)
    axes = [
        figure.add_subplot(density_grid[0, 0]),
        figure.add_subplot(density_grid[0, 1]),
        figure.add_subplot(outer_grid[0, 1]),
    ]
    for axis, quantity, title in zip(
        axes[:2],
        ("reference", "prediction"),
        ("MD", "cDFT"),
    ):
        axis.imshow(images[quantity])
        axis.set_axis_off()
        axis.set_title(title, pad=2)

    parity = axes[2]
    reference = field["reference"].ravel()
    prediction = field["prediction"].ravel()
    wall_distance = minimum_wall_distance(field["external"]).ravel()
    point_norm = Normalize(vmin=0.0, vmax=float(np.max(wall_distance)))
    points = parity.scatter(
        reference,
        prediction,
        c=wall_distance,
        cmap="viridis",
        norm=point_norm,
        s=8,
        alpha=0.58,
        linewidths=0.0,
        rasterized=True,
    )
    parity.plot((0.0, 0.8), (0.0, 0.8), color="0.15", linewidth=1.1, zorder=3)
    parity.set_xlim(0.0, 0.8)
    parity.set_ylim(0.0, 0.8)
    parity.set_aspect("equal", adjustable="box")
    parity.set_xlabel(r"$\rho_{\mathrm{MD}}\sigma^3$")
    parity.set_ylabel(r"$\rho_{\mathrm{cDFT}}\sigma^3$")
    parity.grid(False)
    parity.text(
        0.96,
        0.04,
        "RMSE = {:.3f}\n$R^2$ = {:.3f}".format(field["rmse"], field["r2"]),
        transform=parity.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_SIZE,
    )

    # Scale the scientific panels relative to the preceding production layout.
    # The two contour panels remain centered as a contiguous pair, whereas the
    # parity panel grows leftward and upward so its outer page margins and axis
    # labels remain unchanged.
    figure.canvas.draw()
    contour_boxes = [axis.get_position() for axis in axes[:2]]
    contour_center_x = 0.5 * (contour_boxes[0].x0 + contour_boxes[1].x1)
    contour_center_y = 0.5 * (contour_boxes[0].y0 + contour_boxes[0].y1)
    contour_width = CONTOUR_PANEL_SCALE * contour_boxes[0].width
    contour_height = CONTOUR_PANEL_SCALE * contour_boxes[0].height
    contour_left = contour_center_x - contour_width
    contour_bottom = (
        contour_center_y
        - 0.5 * contour_height
        + LEFT_BLOCK_VERTICAL_SHIFT
        + CONTOUR_GAP_ADJUSTMENT
    )
    axes[0].set_position(
        (contour_left, contour_bottom, contour_width, contour_height)
    )
    axes[1].set_position(
        (contour_left + contour_width, contour_bottom, contour_width, contour_height)
    )

    parity_box = parity.get_position()
    parity_width = PARITY_PANEL_SCALE * parity_box.width
    parity_height = PARITY_PANEL_SCALE * parity_box.height
    parity.set_position(
        (
            parity_box.x1 - parity_width,
            parity_box.y0,
            parity_width,
            parity_height,
        )
    )

    # Bold lower-case panel labels follow the reading order of the combined
    # figure. The parity label uses the open corner above its inset colorbar.
    for axis, label in zip(axes[:2], ("a", "b")):
        axis.text(
            0.02,
            1.03,
            label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
            clip_on=False,
        )
    parity.text(
        -0.22,
        1.02,
        "c",
        transform=parity.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        clip_on=False,
        zorder=5,
    )

    mean_density = float(np.mean(reference))
    mean_density_label = (
        rf"$\langle\rho\rangle\sigma^3 = {mean_density:.2f}$"
    )
    figure.text(
        contour_center_x,
        contour_bottom - 0.025,
        mean_density_label,
        ha="center",
        va="top",
        fontsize=FONT_SIZE,
    )
    parity.text(
        0.08,
        0.72,
        mean_density_label,
        transform=parity.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        zorder=5,
    )

    # Keep the common density scale above the two three-dimensional panels so
    # that the lower edge of the figure remains free of detached legends.
    contour_left = axes[0].get_position().x0
    contour_right = axes[1].get_position().x1
    contour_span = contour_right - contour_left
    density_axis = figure.add_axes(
        (
            contour_left + 0.04 * contour_span,
            0.905 + LEFT_BLOCK_VERTICAL_SHIFT,
            0.92 * contour_span,
            0.028,
        )
    )
    density_bar = figure.colorbar(
        ScalarMappable(norm=DENSITY_NORM, cmap=DENSITY_CMAP),
        cax=density_axis,
        orientation="horizontal",
        ticks=DENSITY_LEVELS,
    )
    density_bar.set_label(r"$\rho\sigma^3$", labelpad=1)
    density_bar.ax.xaxis.set_label_position("top")
    density_bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    density_bar.ax.tick_params(labelsize=FONT_SIZE)

    # The wall-distance scale fits in the otherwise empty region above the
    # parity diagonal.  Keeping it inside the parity panel also makes the
    # geometric meaning of the point colors immediate.
    field_axis = parity.inset_axes((0.08, 0.84, 0.56, 0.035))
    field_bar = figure.colorbar(
        points,
        cax=field_axis,
        orientation="horizontal",
    )
    field_bar.set_ticks((0.0, 1.0, 2.0, 3.0))
    field_bar.outline.set_visible(False)
    field_bar.ax.set_title(r"$d_{\mathrm{wall}}/\sigma$", fontsize=FONT_SIZE, pad=1)
    field_bar.ax.tick_params(labelsize=FONT_SIZE, length=2.5, pad=1)

    stem = RESULTS_DIRECTORY / OUTPUT_STEMS[system]
    figure.savefig(stem.with_suffix(".png"), dpi=400)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def main():
    global RESULTS_DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_directory",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[2] / "output",
        help="output directory",
    )
    parser.add_argument("--system", choices=tuple(OUTPUT_STEMS), default="gyroid-pore")
    parser.add_argument("--panel", nargs=2, metavar=("QUANTITY", "OUTPUT"))
    arguments = parser.parse_args()
    RESULTS_DIRECTORY = arguments.results_directory.expanduser().resolve()
    density_renderer.DIRECTORY = RESULTS_DIRECTORY
    if arguments.panel is not None:
        render_panel(arguments.system, arguments.panel[0], Path(arguments.panel[1]))
        return
    assemble_figure(arguments.system, render_images(arguments.system))


if __name__ == "__main__":
    main()
