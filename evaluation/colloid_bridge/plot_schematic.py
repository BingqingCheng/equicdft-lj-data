#!/usr/bin/env python3
"""Draw the two-colloid geometry and its colloid--fluid potential."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.patches import Patch


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = DIRECTORY.parents[1] / "output"

# Parameters used by the original soft two-colloid reference campaign.
CORE_RADIUS = 2.6
CORE_HEIGHT = 8.0
CORE_WIDTH = 0.50
SHELL_RADIUS = 3.4
SHELL_WIDTH = 0.80
ATTRACTION = 1.4
SEPARATION = 7.0

FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_CM = 7.3
FONT_SIZE_PT = 11.0

COLLOID_COLOR = "#BDBDBD"
COLLOID_EDGE = "#202020"
FLUID_COLOR = "#009E73"
SHELL_COLOR = "#56B4E9"
FORCE_COLOR = "#A52A2A"


def colloid_fluid_potential(distance):
    """Return the logistic core, Gaussian shell, and their sum."""

    repulsive = CORE_HEIGHT / (
        1.0 + np.exp((distance - CORE_RADIUS) / CORE_WIDTH)
    )
    attractive = -ATTRACTION * np.exp(
        -(distance - SHELL_RADIUS) ** 2 / (2.0 * SHELL_WIDTH**2)
    )
    return repulsive, attractive, repulsive + attractive


def fluid_particle_positions(number=74, seed=72841):
    """Generate a reproducible, non-overlapping schematic fluid arrangement."""

    generator = np.random.default_rng(seed)
    centers = np.asarray(((-0.5 * SEPARATION, 0.0), (0.5 * SEPARATION, 0.0)))
    positions = []
    attempts = 0
    while len(positions) < number and attempts < 200000:
        attempts += 1
        candidate = np.asarray(
            (generator.uniform(-8.0, 8.0), generator.uniform(-5.35, 5.35))
        )
        if np.min(np.linalg.norm(candidate[None, :] - centers, axis=1)) < CORE_RADIUS + 0.28:
            continue
        if positions:
            distances = np.linalg.norm(np.asarray(positions) - candidate, axis=1)
            if float(np.min(distances)) < 0.58:
                continue
        positions.append(candidate)
    if len(positions) != number:
        raise RuntimeError("could not generate the requested schematic particles")
    return np.asarray(positions)


def draw_schematic(axis, panel_label=None):
    """Draw two fixed colloids immersed in a one-component LJ fluid."""

    centers = (-0.5 * SEPARATION, 0.5 * SEPARATION)
    for x_position in centers:
        # The thin contour marks where the colloid--LJ attraction is centered;
        # it is not a direct interaction between the two colloids.
        axis.add_patch(
            Circle(
                (x_position, 0.0),
                SHELL_RADIUS,
                fill=False,
                edgecolor=SHELL_COLOR,
                linewidth=1.35,
                alpha=0.62,
                zorder=1,
            )
        )
        axis.add_patch(
            Circle(
                (x_position, 0.0),
                CORE_RADIUS,
                facecolor=COLLOID_COLOR,
                edgecolor=COLLOID_EDGE,
                linewidth=1.2,
                zorder=4,
            )
        )
        axis.plot(x_position, 0.0, marker=".", color=COLLOID_EDGE, markersize=3.0, zorder=5)

    particles = fluid_particle_positions()
    axis.scatter(
        particles[:, 0],
        particles[:, 1],
        s=13.0,
        facecolor=FLUID_COLOR,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.88,
        zorder=3,
    )

    separation_arrow = {
        "arrowstyle": "<->",
        "color": COLLOID_EDGE,
        "linewidth": 1.0,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
        "mutation_scale": 9.0,
    }
    axis.annotate(
        "",
        xy=(centers[1], 4.65),
        xytext=(centers[0], 4.65),
        arrowprops=separation_arrow,
    )
    axis.text(0.0, 4.88, r"$D$", ha="center", va="bottom")

    force_arrow = {
        "arrowstyle": "->",
        "color": FORCE_COLOR,
        "linewidth": 1.2,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
        "mutation_scale": 10.0,
    }
    axis.annotate(
        "",
        xy=(centers[0] - CORE_RADIUS - 1.6, 0.0),
        xytext=(centers[0], 0.0),
        arrowprops=force_arrow,
        zorder=7,
    )
    axis.annotate(
        "",
        xy=(centers[1] + CORE_RADIUS + 1.6, 0.0),
        xytext=(centers[1], 0.0),
        arrowprops=force_arrow,
        zorder=7,
    )
    axis.text(
        centers[0] - CORE_RADIUS - 1.05,
        0.62,
        r"$\langle F\rangle_D$",
        color=FORCE_COLOR,
        ha="center",
        fontsize=10.0,
        zorder=7,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.5},
    )
    axis.text(
        centers[1] + CORE_RADIUS + 1.05,
        0.62,
        r"$\langle F\rangle_D$",
        color=FORCE_COLOR,
        ha="center",
        fontsize=10.0,
        zorder=7,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.5},
    )

    # Relate the schematic directly to the radial coordinate in panel (b).
    angle = np.deg2rad(58.0)
    highlighted_distance = 4.45
    highlighted = np.asarray(
        (
            centers[1] + highlighted_distance * np.cos(angle),
            highlighted_distance * np.sin(angle),
        )
    )
    axis.plot(
        (centers[1], highlighted[0]),
        (0.0, highlighted[1]),
        color="0.20",
        linewidth=0.8,
        zorder=5,
    )
    axis.scatter(
        highlighted[0],
        highlighted[1],
        s=26.0,
        facecolor=FLUID_COLOR,
        edgecolor=COLLOID_EDGE,
        linewidth=0.6,
        zorder=6,
    )
    radial_origin = np.asarray((centers[1], 0.0))
    radial_label = radial_origin + 0.70 * (highlighted - radial_origin)
    axis.text(
        radial_label[0] + 0.14,
        radial_label[1] - 0.12,
        r"$r$",
        ha="left",
        va="center",
        zorder=7,
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.5},
    )

    axis.set_xlim(-8.2, 8.2)
    # Reserve a clean band below the illustrated fluid for the two-row legend.
    axis.set_ylim(-9.0, 5.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    if panel_label is not None:
        axis.text(
            -0.040,
            0.975,
            panel_label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
            clip_on=False,
            zorder=20,
        )
    fluid_handle = Line2D(
        (),
        (),
        marker="o",
        linestyle="none",
        markersize=5.0,
        markerfacecolor=FLUID_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.35,
        label="LJ particle",
    )
    colloid_handle = Patch(
        facecolor=COLLOID_COLOR,
        edgecolor=COLLOID_EDGE,
        linewidth=0.8,
        label="Colloid",
    )
    shell_handle = Line2D(
        (),
        (),
        marker="o",
        linestyle="none",
        markersize=6.0,
        markerfacecolor="none",
        markeredgecolor=SHELL_COLOR,
        markeredgewidth=1.1,
        alpha=0.72,
        label="Colloid–LJ attractive region",
    )
    particle_legend = axis.legend(
        handles=(fluid_handle, colloid_handle),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.120),
        ncol=2,
        fontsize=10.0,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        borderaxespad=0.0,
        borderpad=0.15,
        handlelength=0.85,
        handletextpad=0.25,
        labelspacing=0.0,
        columnspacing=0.45,
    )
    axis.add_artist(particle_legend)
    axis.legend(
        handles=(shell_handle,),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.005),
        ncol=1,
        fontsize=10.0,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        borderaxespad=0.0,
        borderpad=0.15,
        handlelength=0.85,
        handletextpad=0.25,
    )


def draw_potential(axis):
    """Plot the radial colloid--LJ interaction used in the simulations."""

    distance = np.linspace(0.0, 7.0, 1400)
    _, _, total = colloid_fluid_potential(distance)
    axis.plot(distance, total, color=COLLOID_EDGE, linewidth=2.0)
    axis.fill_between(
        distance,
        total,
        0.0,
        where=total < 0.0,
        color=SHELL_COLOR,
        alpha=0.18,
        linewidth=0.0,
    )
    axis.axhline(0.0, color="0.45", linewidth=0.7)

    axis.set_xlim(0.0, 7.0)
    axis.set_ylim(-1.45, 8.25)
    axis.set_xlabel(r"$r_{\mathrm{C-LJ}}/\sigma$")
    axis.set_ylabel(r"$\phi(r)/\epsilon$")
    axis.tick_params(direction="in", top=True, right=True, length=3.0, width=0.8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def main():
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
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(FIGURE_WIDTH_CM * centimeters, FIGURE_HEIGHT_CM * centimeters),
            gridspec_kw={"width_ratios": (1.10, 1.0)},
        )
        draw_schematic(axes[0])
        draw_potential(axes[1])
        figure.subplots_adjust(left=0.015, right=0.985, bottom=0.19, top=0.97, wspace=0.13)

        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIRECTORY / "colloid_schematic_and_potential"
        figure.savefig(output.with_suffix(".png"), dpi=600)
        figure.savefig(output.with_suffix(".pdf"))
        figure.savefig(output.with_suffix(".svg"))
        plt.close(figure)


if __name__ == "__main__":
    main()
