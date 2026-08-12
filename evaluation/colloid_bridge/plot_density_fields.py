#!/usr/bin/env python3
"""Compare MD and cDFT density fields across colloid separations.

Each displayed field is averaged over the 1-sigma-thick slab centered on the
plane containing the colloid centers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APPLICATION_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = APPLICATION_DIRECTORY.parents[1]
SOURCE_DATA = REPOSITORY / "source_data" / "fig03_colloid_bridge" / "density_fields.npz"
OUTPUT_DIRECTORY = REPOSITORY / "output"
SEPARATIONS = (5.0, 6.5, 9.0)
SLAB_WIDTH = 1.0
EXTERNAL_LEVEL = 4.0
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 8.8
FONT_SIZE_PT = 10.0
DENSITY_LIMIT = 0.65
X_LIMIT = (-8.0, 8.0)
Y_LIMIT = (-5.5, 5.5)


def load_fields(separation):
    """Load aligned reference, prediction, and external-potential grids."""

    key = str(float(separation)).replace(".", "p")
    with np.load(SOURCE_DATA) as data:
        reference = data[f"reference_D{key}"]
        prediction = data[f"prediction_D{key}"]
        external = data[f"external_D{key}"]
        grid_spacing = data[f"spacing_D{key}"]
    grid_size = np.asarray(reference.shape, dtype=int)
    box_lengths = grid_spacing * grid_size
    center = 0.5 * box_lengths
    coordinates = tuple(
        (np.arange(size, dtype=float) + 0.5) * spacing - midpoint
        for size, spacing, midpoint in zip(grid_size, grid_spacing, center)
    )
    slab_indices = np.flatnonzero(np.abs(coordinates[2]) <= 0.5 * SLAB_WIDTH)
    if len(slab_indices) == 0:
        raise ValueError("the requested central slab contains no grid planes")

    def slab_average(grid):
        return np.mean(grid[:, :, slab_indices], axis=2).T

    extent = (
        coordinates[0][0] - 0.5 * grid_spacing[0],
        coordinates[0][-1] + 0.5 * grid_spacing[0],
        coordinates[1][0] - 0.5 * grid_spacing[1],
        coordinates[1][-1] + 0.5 * grid_spacing[1],
    )
    return {
        "reference": slab_average(reference),
        "prediction": slab_average(prediction),
        "external": slab_average(external),
        "extent": extent,
    }


def draw_panel(axis, field, quantity, separation):
    """Draw one density field with the effective colloid wall."""

    image = axis.imshow(
        field[quantity],
        origin="lower",
        extent=field["extent"],
        cmap="viridis",
        vmin=0.0,
        vmax=DENSITY_LIMIT,
        interpolation="bilinear",
        rasterized=True,
    )
    boundary_color = "white"

    axis.contour(
        field["external"],
        levels=(EXTERNAL_LEVEL,),
        origin="lower",
        extent=field["extent"],
        colors=boundary_color,
        linewidths=0.75,
    )
    axis.scatter(
        (-0.5 * separation, 0.5 * separation),
        (0.0, 0.0),
        marker=".",
        s=7.0,
        color=boundary_color,
        linewidths=0.0,
        zorder=3,
    )
    axis.set_xlim(*X_LIMIT)
    axis.set_ylim(*Y_LIMIT)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(direction="in", top=True, right=True, length=2.5, width=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
    return image


def draw_density_comparison(figure, subplot_spec, fields=None, panel_label=None):
    """Draw the complete two-row density comparison into ``subplot_spec``.

    The color bar is a fourth, narrow column and therefore remains part of the
    density subfigure in both the standalone and combined figures.
    """

    if fields is None:
        fields = [load_fields(separation) for separation in SEPARATIONS]
    grid = subplot_spec.subgridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        wspace=0.07,
        hspace=0.08,
    )
    axes = [
        [figure.add_subplot(grid[row, column]) for column in range(3)]
        for row in range(2)
    ]

    density_image = None
    for column, (separation, field) in enumerate(zip(SEPARATIONS, fields)):
        density_image = draw_panel(axes[0][column], field, "reference", separation)
        draw_panel(axes[1][column], field, "prediction", separation)
        axes[0][column].set_title(
            r"$D/\sigma={:.1f}$".format(separation),
            pad=3.0,
        )

    axes[0][0].text(
        0.035,
        0.94,
        r"$\rho_{\rm MD}$",
        transform=axes[0][0].transAxes,
        ha="left",
        va="top",
        color="white",
        bbox={"facecolor": "0.10", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
    )
    axes[1][0].text(
        0.035,
        0.94,
        r"$\rho_{\rm cDFT}$",
        transform=axes[1][0].transAxes,
        ha="left",
        va="top",
        color="white",
        bbox={"facecolor": "0.10", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
    )
    if panel_label is not None:
        axes[0][0].text(
            -0.17,
            1.13,
            panel_label,
            transform=axes[0][0].transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )

    for row in range(2):
        axes[row][0].set_ylabel(r"$y/\sigma$", labelpad=1.0)
        for column in (1, 2):
            axes[row][column].tick_params(labelleft=False)
    for column in range(3):
        axes[0][column].tick_params(labelbottom=False)
        axes[1][column].set_xlabel(r"$x/\sigma$", labelpad=1.0)

    colorbar_axis = figure.add_subplot(grid[:, 3])
    colorbar = figure.colorbar(
        density_image,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=(0.0, 0.2, 0.4, 0.6),
    )
    colorbar.set_label(r"Number density, $\rho\sigma^3$", labelpad=3.0)
    colorbar.ax.tick_params(direction="in", length=2.5, width=0.7, pad=2.0)
    colorbar.outline.set_linewidth(0.7)
    if colorbar.vmax != DENSITY_LIMIT:
        raise RuntimeError("density color scale changed unexpectedly")
    return axes, colorbar


def main():
    centimeters = 1.0 / 2.54
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT + 1.0,
        "xtick.labelsize": FONT_SIZE_PT - 1.0,
        "ytick.labelsize": FONT_SIZE_PT - 1.0,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        figure = plt.figure(
            figsize=(FIGURE_WIDTH_CM * centimeters, FIGURE_HEIGHT_CM * centimeters),
        )
        layout = figure.add_gridspec(
            1,
            1,
            left=0.075,
            right=0.94,
            bottom=0.15,
            top=0.93,
        )[0]
        draw_density_comparison(figure, layout)
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIRECTORY / "density_field_comparison"
        figure.savefig(output.with_suffix(".png"), dpi=600)
        figure.savefig(output.with_suffix(".pdf"))
        figure.savefig(output.with_suffix(".svg"))
        plt.close(figure)


if __name__ == "__main__":
    main()
