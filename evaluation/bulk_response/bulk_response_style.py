"""Shared visual encoding for the bulk-response figure."""

import matplotlib.colors as mcolors
import numpy as np


MD_COLOR = "#3F4650"
CDFT_COLOR = "#C84A3A"

# Muted temperature anchors on one ordered scale spanning all four panels.
_TEMPERATURE_ANCHORS = np.asarray(
    [0.65, 0.80, 0.95, 1.00, 1.05, 1.20, 1.50, 1.80], dtype=float
)
_TEMPERATURE_RGB = np.asarray(
    [
        mcolors.to_rgb("#42418B"),
        mcolors.to_rgb("#477CD6"),
        mcolors.to_rgb("#25B4B5"),
        mcolors.to_rgb("#24C1A3"),
        mcolors.to_rgb("#31CB8E"),
        mcolors.to_rgb("#85D746"),
        mcolors.to_rgb("#D78C35"),
        mcolors.to_rgb("#952212"),
    ],
    dtype=float,
)


def temperature_color(temperature):
    """Return the shared figure color for one reduced temperature."""

    value = float(temperature)
    rgb = [
        np.interp(value, _TEMPERATURE_ANCHORS, _TEMPERATURE_RGB[:, channel])
        for channel in range(3)
    ]
    return mcolors.to_hex(rgb)
