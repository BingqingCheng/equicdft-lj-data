#!/usr/bin/env python3
"""Plot direct and density-based two-colloid mean forces."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator


APPLICATION_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = APPLICATION_DIRECTORY.parents[1]
DIRECTORY = REPOSITORY / "source_data" / "fig03_colloid_bridge"
OUTPUT_DIRECTORY = REPOSITORY / "output"
FIGURE_WIDTH_CM = 8.0
FIGURE_HEIGHT_CM = 5.5
FONT_SIZE_PT = 11.0
FORCE_LEGEND_LABELS = (
    r"Direct MD",
    r"HF: $\rho_{\rm MD}$",
    r"HF: $\rho_{\rm cDFT}$",
)


def load_force_data():
    """Load the discrete MD estimates and dense cDFT force curve."""

    data = pd.read_csv(DIRECTORY / "hellmann_feynman_force.csv")
    dense_cdft = pd.read_csv(DIRECTORY / "cdft_force_dense.csv")
    return data, dense_cdft


def add_colloid_inset(axis):
    """Draw the separation coordinate and positive-force convention."""

    # Keep the schematic large enough to remain legible at the final 8 cm
    # figure width, and place it away from the upper part of the force curve.
    inset = axis.inset_axes([0.67, 0.035, 0.30, 0.55], zorder=10)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.set_aspect("equal", adjustable="box")
    inset.set_facecolor((1.0, 1.0, 1.0, 0.82))
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)

    centers = (0.30, 0.70)
    for center in centers:
        inset.add_patch(
            Circle(
                (center, 0.38),
                0.13,
                facecolor="0.78",
                edgecolor="0.12",
                linewidth=0.8,
                zorder=2,
            )
        )
        inset.plot(center, 0.38, marker=".", color="0.12", markersize=2.2)

    arrow = dict(
        arrowstyle="<->",
        color="0.12",
        linewidth=0.9,
        shrinkA=0.0,
        shrinkB=0.0,
        mutation_scale=7.0,
    )
    inset.annotate("", xy=(centers[1], 0.84), xytext=(centers[0], 0.84), arrowprops=arrow)
    inset.text(0.50, 0.68, r"$D$", ha="center", va="center")

    force_arrow = dict(
        arrowstyle="->",
        color="#A52A2A",
        linewidth=0.9,
        shrinkA=0.0,
        shrinkB=0.0,
        mutation_scale=7.0,
    )
    inset.annotate("", xy=(0.02, 0.38), xytext=(0.16, 0.38), arrowprops=force_arrow)
    inset.annotate("", xy=(0.98, 0.38), xytext=(0.84, 0.38), arrowprops=force_arrow)
    inset.text(
        0.08,
        0.54,
        r"$\langle F\rangle_D$",
        color="#A52A2A",
        ha="center",
        va="center",
    )
    inset.text(
        0.92,
        0.54,
        r"$\langle F\rangle_D$",
        color="#A52A2A",
        ha="center",
        va="center",
    )


def draw_force_curve(
    axis,
    data,
    dense_cdft,
    selected_separations=(),
    add_legend=False,
    panel_label=None,
):
    """Draw the three force estimators and return their legend handles.

    ``selected_separations`` marks the separations used in the density panels.
    The direct MD values remain unconnected because they are independent
    simulation estimates.
    """

    axis.axhline(0.0, color="0.55", linewidth=0.6, zorder=0)
    direct_md = axis.errorbar(
        data["separation"],
        data["direct_MD_force"],
        yerr=data["direct_MD_block_sem"],
        color="0.10",
        ecolor="0.10",
        marker="o",
        markersize=3.2,
        markeredgewidth=0.6,
        linestyle="none",
        capsize=2.0,
        capthick=0.8,
        elinewidth=0.8,
        zorder=4,
    )
    hf_md, = axis.plot(
        data["separation"],
        data["HF_force_MD_density"],
        color="#D55E00",
        marker="s",
        markersize=3.0,
        markeredgewidth=0.5,
        linestyle="none",
        zorder=2,
    )
    hf_cdft, = axis.plot(
        dense_cdft["separation"],
        dense_cdft["HF_force_cDFT_density"],
        color="#0072B2",
        marker="o",
        markersize=1.5,
        markeredgewidth=0.0,
        linewidth=1.1,
        zorder=3,
    )

    # Link the force curve visually to the three density-map columns without
    # implying additional simulation data at those separations.
    if selected_separations:
        for separation in selected_separations:
            axis.axvline(
                separation,
                color="0.82",
                linewidth=0.55,
                linestyle=(0, (2.0, 2.0)),
                zorder=-1,
            )

    all_force_values = pd.concat(
        [
            dense_cdft["HF_force_cDFT_density"],
            data["HF_force_MD_density"],
            data["direct_MD_force"] - data["direct_MD_block_sem"],
            data["direct_MD_force"] + data["direct_MD_block_sem"],
        ],
        ignore_index=True,
    )
    force_min = float(all_force_values.min())
    force_max = float(all_force_values.max())
    force_padding = 0.07 * (force_max - force_min)
    axis.set_xlim(4.35, 10.15)
    axis.set_ylim(force_min - force_padding, force_max + force_padding)
    axis.set_xticks([5, 6, 7, 8, 9, 10])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axis.set_xlabel(r"$D/\sigma$", labelpad=1.0)
    axis.set_ylabel(r"$\langle F\rangle_D/(\epsilon/\sigma)$", labelpad=1.0)
    axis.tick_params(
        direction="in",
        length=2.5,
        width=0.7,
        pad=2.0,
        top=True,
        right=True,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
    if panel_label is not None:
        axis.text(
            -0.040,
            0.965,
            panel_label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )
    handles = (direct_md, hf_md, hf_cdft)
    if add_legend:
        axis.legend(
            handles,
            FORCE_LEGEND_LABELS,
            loc="lower right",
            frameon=False,
            borderpad=0.0,
            borderaxespad=0.35,
            handlelength=1.05,
            handletextpad=0.35,
            labelspacing=0.16,
        )
    return handles


def main():
    data, dense_cdft = load_force_data()

    centimeters = 1.0 / 2.54
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
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
        axis = figure.add_axes([0.17, 0.34, 0.81, 0.63])
        direct_md, hf_md, hf_cdft = draw_force_curve(axis, data, dense_cdft)

        add_colloid_inset(axis)
        legend = figure.legend(
            [direct_md, hf_md, hf_cdft],
            FORCE_LEGEND_LABELS,
            loc="lower center",
            bbox_to_anchor=(0.50, 0.015),
            ncol=3,
            frameon=False,
            borderpad=0.0,
            borderaxespad=0.0,
            handlelength=1.0,
            handletextpad=0.30,
            labelspacing=0.05,
            columnspacing=0.75,
        )

        # Guard the deliberately compact layout against clipping or overlap if
        # labels or Matplotlib defaults change later.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer)
        figure_box = figure.bbox
        x_label_box = axis.xaxis.label.get_window_extent(renderer)
        if not (
            figure_box.contains(legend_box.x0, legend_box.y0)
            and figure_box.contains(legend_box.x1, legend_box.y1)
        ):
            raise RuntimeError(
                "legend extends outside the figure: legend={}, figure={}".format(
                    legend_box.bounds,
                    figure_box.bounds,
                )
            )
        if legend_box.overlaps(x_label_box):
            raise RuntimeError("legend overlaps the x-axis label")

        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIRECTORY / "force_vs_separation"
        figure.savefig(output.with_suffix(".png"), dpi=600)
        figure.savefig(output.with_suffix(".pdf"))
        figure.savefig(output.with_suffix(".svg"))
        plt.close(figure)


if __name__ == "__main__":
    main()
