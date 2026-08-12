"""Mean-field continuation used for the liquid-vapor coexistence curve."""

import numpy as np


def fit_critical_model(temperature, vapor, liquid, converged, minimum_gap):
    gap = liquid - vapor
    selected = converged & (gap >= minimum_gap) & (gap <= 0.55)
    if np.count_nonzero(selected) < 4:
        return None
    gap_slope, gap_intercept = np.polyfit(
        temperature[selected], gap[selected] ** 2, deg=1
    )
    if gap_slope >= 0.0:
        return None
    critical_temperature = -gap_intercept / gap_slope
    diameter = 0.5 * (vapor + liquid)
    diameter_slope, diameter_intercept = np.polyfit(
        temperature[selected], diameter[selected], deg=1
    )
    return {
        "critical_temperature": float(critical_temperature),
        "critical_density": float(
            diameter_slope * critical_temperature + diameter_intercept
        ),
        "gap_slope": float(gap_slope),
        "gap_intercept": float(gap_intercept),
        "diameter_slope": float(diameter_slope),
        "diameter_intercept": float(diameter_intercept),
        "maximum_fitted_temperature": float(np.max(temperature[selected])),
    }
