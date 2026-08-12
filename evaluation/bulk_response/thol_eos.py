"""Thol et al. Helmholtz EOS for the LJTS fluid with ``r_c = 2.5 sigma``.

The 21 coefficients are from M. Thol et al., Int. J. Thermophys. 36, 25
(2015), DOI: 10.1007/s10765-014-1764-4. All inputs and outputs use reduced
Lennard-Jones units.
"""

import numpy as np


CRITICAL_TEMPERATURE = 1.086
CRITICAL_DENSITY = 0.319
CRITICAL_PRESSURE = 0.101

# Columns: term type, n, t, d, l, eta, beta, gamma, epsilon.
# Type 1 is polynomial, type 2 has exp(-delta**l), and type 3 is Gaussian.
COEFFICIENTS = np.asarray(
    [
        [1, 0.0156060840, 1.000, 4, 0, 0, 0, 0, 0],
        [1, 1.7917527000, 0.304, 1, 0, 0, 0, 0, 0],
        [1, -1.9613228000, 0.583, 1, 0, 0, 0, 0, 0],
        [1, 1.3045604000, 0.662, 2, 0, 0, 0, 0, 0],
        [1, -1.8117673000, 0.870, 2, 0, 0, 0, 0, 0],
        [1, 0.1548399700, 0.870, 3, 0, 0, 0, 0, 0],
        [2, -0.4751098200, 1.286, 1, 1, 0, 0, 0, 0],
        [2, -0.5842280700, 1.960, 1, 2, 0, 0, 0, 0],
        [2, -0.5060736400, 2.400, 3, 2, 0, 0, 0, 0],
        [2, 0.1163964400, 1.700, 2, 1, 0, 0, 0, 0],
        [2, -0.2009241200, 3.000, 2, 2, 0, 0, 0, 0],
        [2, -0.0948852040, 1.250, 5, 1, 0, 0, 0, 0],
        [3, 0.0094333106, 3.600, 1, 0, 4.70, 20.0, 1.0, 0.55],
        [3, 0.3044462800, 2.080, 1, 0, 1.92, 0.77, 0.5, 0.70],
        [3, -0.0010820946, 5.240, 2, 0, 2.70, 0.50, 0.8, 2.00],
        [3, -0.0996933910, 0.960, 3, 0, 1.49, 0.80, 1.5, 1.14],
        [3, 0.0091193522, 1.360, 3, 0, 0.65, 0.40, 0.7, 1.20],
        [3, 0.1297054300, 1.655, 2, 0, 1.73, 0.43, 1.6, 1.31],
        [3, 0.0230360300, 0.900, 1, 0, 3.70, 8.00, 1.3, 1.14],
        [3, -0.0826710730, 0.860, 2, 0, 1.90, 3.30, 0.6, 0.53],
        [3, -2.2497821000, 3.950, 3, 0, 13.2, 114.0, 1.3, 0.96],
    ],
    dtype=float,
)


def residual_helmholtz_derivatives(temperature, density):
    """Return ``alpha_r``, ``delta alpha_r_delta``, and
    ``delta^2 alpha_r_deltadelta``.

    ``alpha_r = A_res / (N k_B T)`` is the residual dimensionless Helmholtz
    energy. Temperature and density may be scalars or broadcastable arrays.
    """

    temperature, density = np.broadcast_arrays(
        np.asarray(temperature, dtype=float),
        np.asarray(density, dtype=float),
    )
    if np.any(temperature <= 0.0) or np.any(density < 0.0):
        raise ValueError("temperature must be positive and density nonnegative")

    tau = CRITICAL_TEMPERATURE / temperature
    delta = density / CRITICAL_DENSITY
    alpha_r = np.zeros_like(tau)
    alpha_r_delta = np.zeros_like(tau)
    alpha_r_deltadelta = np.zeros_like(tau)

    for row in COEFFICIENTS:
        term_type, n, t, d, exponent, eta, beta, gamma, epsilon = row
        term = n * tau**t * delta**d

        if term_type == 1:
            logarithmic_derivative = d
            derivative_of_logarithmic_derivative = 0.0
        elif term_type == 2:
            delta_power = delta**exponent
            term = term * np.exp(-delta_power)
            logarithmic_derivative = d - exponent * delta_power
            derivative_of_logarithmic_derivative = -(exponent**2) * delta_power
        else:
            term = term * np.exp(
                -eta * (delta - epsilon) ** 2
                -beta * (tau - gamma) ** 2
            )
            logarithmic_derivative = d - 2.0 * eta * delta * (
                delta - epsilon
            )
            derivative_of_logarithmic_derivative = (
                -2.0 * eta * delta * (2.0 * delta - epsilon)
            )

        alpha_r += term
        alpha_r_delta += term * logarithmic_derivative
        alpha_r_deltadelta += term * (
            logarithmic_derivative**2
            + derivative_of_logarithmic_derivative
            - logarithmic_derivative
        )

    return alpha_r, alpha_r_delta, alpha_r_deltadelta


def evaluate(temperature, density):
    """Evaluate the homogeneous EOS in reduced Lennard-Jones units."""

    temperature, density = np.broadcast_arrays(
        np.asarray(temperature, dtype=float),
        np.asarray(density, dtype=float),
    )
    alpha_r, a01, a02 = residual_helmholtz_derivatives(
        temperature,
        density,
    )
    pressure = density * temperature * (1.0 + a01)
    return {
        "temperature": temperature,
        "density": density,
        "pressure": pressure,
        "beta_pressure": pressure / temperature,
        "alpha_residual": alpha_r,
        "beta_mu_residual": alpha_r + a01,
        "dp_drho": temperature * (1.0 + 2.0 * a01 + a02),
    }


def coexistence_densities(temperature):
    """Return the vapor and liquid coexistence densities below ``T_c``.

    The two densities have equal pressure and chemical potential according to
    the Thol EOS. The ideal-gas thermal constant cancels from the chemical-
    potential equality, leaving ``log(rho) + beta_mu_residual``.
    """

    from scipy.optimize import least_squares

    temperature = float(temperature)
    if not 0.0 < temperature < CRITICAL_TEMPERATURE:
        raise ValueError("coexistence requires 0 < temperature < T_c")

    order_parameter = np.sqrt(1.0 - temperature / CRITICAL_TEMPERATURE)
    initial_vapor = CRITICAL_DENSITY * np.exp(-4.5 * order_parameter)
    initial_liquid = CRITICAL_DENSITY + 0.8 * order_parameter

    def residual(log_density):
        density = np.exp(log_density)
        values = evaluate(temperature, density)
        beta_mu = np.log(density) + values["beta_mu_residual"]
        return np.asarray(
            [
                (values["pressure"][0] - values["pressure"][1])
                / temperature,
                beta_mu[0] - beta_mu[1],
            ]
        )

    solution = least_squares(
        residual,
        np.log([initial_vapor, initial_liquid]),
        bounds=(
            np.log([1.0e-10, CRITICAL_DENSITY * (1.0 + 1.0e-8)]),
            np.log([CRITICAL_DENSITY * (1.0 - 1.0e-8), 1.4]),
        ),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=2000,
    )
    if not solution.success or np.linalg.norm(residual(solution.x)) > 1.0e-8:
        raise RuntimeError("Thol EOS coexistence solve did not converge")
    vapor, liquid = np.exp(solution.x)
    return float(vapor), float(liquid)
