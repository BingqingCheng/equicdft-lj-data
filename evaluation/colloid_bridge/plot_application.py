#!/usr/bin/env python3
"""Assemble the two-colloid geometry, force, and density comparisons."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_schematic import draw_schematic
from plot_density_fields import SEPARATIONS, draw_density_comparison
from plot_force import draw_force_curve, load_force_data


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = DIRECTORY.parents[1] / "output"
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 13.6
FONT_SIZE_PT = 11.0
SCHEMATIC_VERTICAL_SHIFT_CM = 0.5


def main():
    """Render one paper-width summary of the colloid-bridging application."""

    centimeters = 1.0 / 2.54
    force_data, dense_cdft = load_force_data()
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }

    with plt.rc_context(style):
        figure = plt.figure(
            figsize=(FIGURE_WIDTH_CM * centimeters, FIGURE_HEIGHT_CM * centimeters)
        )
        outer = figure.add_gridspec(
            2,
            1,
            height_ratios=(0.82, 1.38),
            left=0.075,
            right=0.910,
            bottom=0.105,
            top=0.985,
            hspace=0.24,
        )
        upper = outer[0].subgridspec(
            1,
            2,
            width_ratios=(1.00, 1.65),
            wspace=0.28,
        )

        schematic_axis = figure.add_subplot(upper[0, 0])
        force_axis = figure.add_subplot(upper[0, 1])
        schematic_position = schematic_axis.get_position()
        schematic_shift = SCHEMATIC_VERTICAL_SHIFT_CM / FIGURE_HEIGHT_CM
        schematic_axis.set_position(
            (
                schematic_position.x0,
                schematic_position.y0 - schematic_shift,
                schematic_position.width,
                schematic_position.height,
            )
        )
        draw_schematic(schematic_axis, panel_label="a")
        draw_force_curve(
            force_axis,
            force_data,
            dense_cdft,
            selected_separations=SEPARATIONS,
            add_legend=True,
            panel_label="c",
        )
        draw_density_comparison(figure, outer[1], panel_label="b")

        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIRECTORY / "two_colloid_application_figure"
        figure.savefig(output.with_suffix(".png"), dpi=600)
        figure.savefig(output.with_suffix(".pdf"))
        figure.savefig(output.with_suffix(".svg"))
        plt.close(figure)


if __name__ == "__main__":
    main()
