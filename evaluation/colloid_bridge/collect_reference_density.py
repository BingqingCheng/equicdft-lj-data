#!/usr/bin/env python3
"""Combine a LAMMPS density grid with an application external potential."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


APPLICATIONS = (
    "two-colloid-bridging",
    "two-colloid-bridging-deepergaussian",
    "wavy-wall",
    "mickey-mouse",
    "janus-colloid",
    "gyroid-pore",
    "chemically-patterned-wall",
)


def read_density_grid(path):
    """Read the last complete fix ave/chunk block."""

    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    cursor = 0
    last_rows = []
    while cursor < len(lines):
        header = lines[cursor].split()
        if len(header) != 3:
            raise ValueError("malformed ave/chunk header in {}".format(path))
        number_of_bins = int(header[1])
        raw_rows = lines[cursor + 1 : cursor + 1 + number_of_bins]
        if len(raw_rows) != number_of_bins:
            raise ValueError("incomplete density block in {}".format(path))
        rows = []
        for raw_row in raw_rows:
            columns = raw_row.split()
            if len(columns) < 6:
                raise ValueError("malformed density row in {}".format(path))
            rows.append(
                (
                    int(columns[0]),
                    float(columns[1]),
                    float(columns[2]),
                    float(columns[3]),
                    float(columns[5]),
                )
            )
        last_rows = rows
        cursor += number_of_bins + 1
    if not last_rows:
        raise ValueError("no complete density block in {}".format(path))
    return last_rows


def grid_geometry(rows, expected_spacing):
    axes = [sorted(set(row[index] for row in rows)) for index in (1, 2, 3)]
    if any(len(axis) < 2 for axis in axes):
        raise ValueError("the density grid needs at least two cells per axis")
    shape = tuple(len(axis) for axis in axes)
    if math.prod(shape) != len(rows):
        raise ValueError("density rows do not form a complete Cartesian grid")
    spacing = tuple(axis[1] - axis[0] for axis in axes)
    tolerance = 1.0e-8
    for axis, delta in zip(axes, spacing):
        if abs(delta - expected_spacing) > tolerance:
            raise ValueError(
                "expected grid spacing {}, found {}".format(expected_spacing, delta)
            )
        if any(abs((b - a) - delta) > tolerance for a, b in zip(axis, axis[1:])):
            raise ValueError("density grid is not uniform")
    lengths = tuple(count * delta for count, delta in zip(shape, spacing))
    lower = tuple(axis[0] - 0.5 * delta for axis, delta in zip(axes, spacing))
    if any(abs(value) > tolerance for value in lower):
        raise ValueError("application grids must have lower box bounds at zero")
    index_maps = [{value: index for index, value in enumerate(axis)} for axis in axes]
    return axes, shape, spacing, lengths, index_maps


def parse_parameters(values):
    parameters = {}
    for value in values:
        if "=" not in value:
            raise ValueError("parameter must have NAME=VALUE form: {}".format(value))
        name, raw = value.split("=", 1)
        if not name or name in parameters:
            raise ValueError("invalid or repeated parameter: {}".format(name))
        parameters[name] = float(raw)
    return parameters


def require(parameters, *names):
    missing = [name for name in names if name not in parameters]
    if missing:
        raise ValueError("missing parameters: {}".format(", ".join(missing)))
    return tuple(parameters[name] for name in names)


def periodic_delta(value, center, length):
    delta = value - center
    return delta - length * math.floor(delta / length + 0.5)


def make_evaluator(application, parameters, lengths):
    lx, ly, lz = lengths

    if application in (
        "two-colloid-bridging",
        "two-colloid-bridging-deepergaussian",
    ):
        separation, radius, core_height, core_width, shell_radius, shell_width, attraction = require(
            parameters,
            "separation",
            "colloid_radius",
            "core_height",
            "core_width",
            "shell_radius",
            "shell_width",
            "attraction_epsilon",
        )
        centers = (
            (0.5 * lx - 0.5 * separation, 0.5 * ly, 0.5 * lz),
            (0.5 * lx + 0.5 * separation, 0.5 * ly, 0.5 * lz),
        )

        def evaluate(x, y, z):
            potential = 0.0
            for cx, cy, cz in centers:
                dx = periodic_delta(x, cx, lx)
                dy = periodic_delta(y, cy, ly)
                dz = periodic_delta(z, cz, lz)
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                core = core_height / (
                    1.0 + math.exp((distance - radius) / core_width)
                )
                shell = math.exp(
                    -((distance - shell_radius) ** 2)
                    / (2.0 * shell_width * shell_width)
                )
                potential += core - attraction * shell
            return potential

        return evaluate

    if application == "mickey-mouse":
        body_radius, ear_radius, core_height, core_width = require(
            parameters,
            "body_radius",
            "ear_radius",
            "core_height",
            "core_width",
        )
        centers = (
            (0.5 * lx, 0.5 * ly, 0.5 * lz - 1.0, body_radius),
            (0.5 * lx - 2.5, 0.5 * ly, 0.5 * lz + 2.0, ear_radius),
            (0.5 * lx + 2.5, 0.5 * ly, 0.5 * lz + 2.0, ear_radius),
        )

        def evaluate(x, y, z):
            potential = 0.0
            for cx, cy, cz, radius in centers:
                dx = periodic_delta(x, cx, lx)
                dy = periodic_delta(y, cy, ly)
                dz = periodic_delta(z, cz, lz)
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                potential += core_height / (
                    1.0 + math.exp((distance - radius) / core_width)
                )
            return potential

        return evaluate

    if application == "wavy-wall":
        mean_gap, amplitude, height, width, phase = require(
            parameters,
            "mean_gap",
            "wall_amplitude",
            "wall_height",
            "wall_width",
            "phase",
        )
        qx, qy = 2.0 * math.pi / lx, 2.0 * math.pi / ly

        def evaluate(x, y, z):
            zlo = 0.5 * lz - 0.5 * mean_gap + amplitude * math.cos(qx * x) * math.cos(qy * y)
            zhi = 0.5 * lz + 0.5 * mean_gap + amplitude * math.cos(qx * x + phase) * math.cos(qy * y)
            dlo, dhi = z - zlo, zhi - z
            return height / (1.0 + math.exp(dlo / width)) + height / (
                1.0 + math.exp(dhi / width)
            )

        return evaluate

    if application == "janus-colloid":
        radius, core_height, core_width, shell_radius, shell_width, average, difference, angular_width = require(
            parameters,
            "colloid_radius",
            "core_height",
            "core_width",
            "shell_radius",
            "shell_width",
            "epsilon_average",
            "epsilon_difference",
            "angular_width",
        )
        center = (0.5 * lx, 0.5 * ly, 0.5 * lz)

        def evaluate(x, y, z):
            dx = periodic_delta(x, center[0], lx)
            dy = periodic_delta(y, center[1], ly)
            dz = periodic_delta(z, center[2], lz)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz + 1.0e-20)
            cosine = dz / distance
            angular_norm = math.sqrt(cosine * cosine + angular_width * angular_width)
            epsilon = average + difference * cosine / angular_norm
            core = core_height / (1.0 + math.exp((distance - radius) / core_width))
            shell = math.exp(
                -((distance - shell_radius) ** 2) / (2.0 * shell_width * shell_width)
            )
            return core - epsilon * shell

        return evaluate

    if application == "gyroid-pore":
        g0, height, width = require(parameters, "g0", "wall_height", "wall_width")
        q = 2.0 * math.pi / lx
        if abs(ly - lx) > 1.0e-8 or abs(lz - lx) > 1.0e-8:
            raise ValueError("gyroid input requires a cubic box")

        def evaluate(x, y, z):
            g = (
                math.sin(q * x) * math.cos(q * y)
                + math.sin(q * y) * math.cos(q * z)
                + math.sin(q * z) * math.cos(q * x)
            )
            return height / (1.0 + math.exp((g - g0) / width))

        return evaluate

    if application == "chemically-patterned-wall":
        zlower, zupper, height, wall_width, average, modulation, attraction_distance, attraction_width = require(
            parameters,
            "zlower",
            "zupper",
            "wall_height",
            "wall_width",
            "attraction_average",
            "attraction_modulation",
            "attraction_distance",
            "attraction_width",
        )
        qx, qy = 2.0 * math.pi / lx, 2.0 * math.pi / ly

        def evaluate(x, y, z):
            dlo, dhi = z - zlower, zupper - z
            repulsion = height / (1.0 + math.exp(dlo / wall_width)) + height / (
                1.0 + math.exp(dhi / wall_width)
            )
            epsilon = average + modulation * math.cos(qx * x) * math.cos(qy * y)
            attraction = math.exp(
                -((dlo - attraction_distance) ** 2)
                / (2.0 * attraction_width * attraction_width)
            )
            return repulsion - epsilon * attraction

        return evaluate

    raise ValueError("unsupported application: {}".format(application))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("density_grid", type=Path)
    parser.add_argument("--application", choices=APPLICATIONS, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--particle-count", type=int, required=True)
    parser.add_argument("--parameter", action="append", default=[])
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--potential-cap", type=float, default=50.0)
    parser.add_argument(
        "--external-output", type=Path, default=Path("external_potential_grid.dat")
    )
    parser.add_argument(
        "--extxyz-output",
        type=Path,
        default=Path("density_external_grid_0.5sigma.extxyz"),
    )
    arguments = parser.parse_args()

    if arguments.temperature <= 0.0 or arguments.particle_count <= 0:
        raise ValueError("temperature and particle count must be positive")
    if arguments.potential_cap <= 0.0:
        raise ValueError("potential cap must be positive")

    rows = read_density_grid(arguments.density_grid)
    _axes, shape, spacing, lengths, index_maps = grid_geometry(
        rows, arguments.grid_spacing
    )
    parameters = parse_parameters(arguments.parameter)
    evaluate = make_evaluator(arguments.application, parameters, lengths)

    combined = []
    density_integral = 0.0
    for bin_id, x, y, z, density in rows:
        potential = min(evaluate(x, y, z), arguments.potential_cap)
        combined.append((bin_id, x, y, z, density, potential))
        density_integral += density * math.prod(spacing)

    external_lines = [
        "# External potential for {}".format(arguments.application),
        "# V_ext is capped above at {:.16g}".format(arguments.potential_cap),
        "# Bin x y z V_ext",
    ]
    for bin_id, x, y, z, _density, potential in combined:
        external_lines.append(
            "{} {:.16g} {:.16g} {:.16g} {:.16g}".format(
                bin_id, x, y, z, potential
            )
        )
    arguments.external_output.write_text("\n".join(external_lines) + "\n")

    nx, ny, nz = shape
    lx, ly, lz = lengths
    comment = (
        'Lattice="{} 0 0 0 {} 0 0 0 {}" '.format(nx, ny, nz)
        + 'physical_lattice="{:.16g} 0 0 0 {:.16g} 0 0 0 {:.16g}" '.format(
            lx, ly, lz
        )
        + "Properties=species:S:1:pos:R:3:grid_center:R:3:grid_id:I:1:"
        + "density:R:1:V_ext:R:1 "
        + 'pbc="T T T" T={:.16g} mu=nan '.format(arguments.temperature)
        + 'grid_size="{} {} {}" '.format(nx, ny, nz)
        + 'grid_spacing="{:.16g} {:.16g} {:.16g}" '.format(*spacing)
        + 'grid_indexing="zero_based" ensemble=NVT N={} '.format(
            arguments.particle_count
        )
        + 'application="{}" V_ext_cap={:.16g}'.format(
            arguments.application, arguments.potential_cap
        )
    )
    # Keep the exact analytic-field parameters in every generated data frame.
    # This makes otherwise similar application campaigns self-describing.
    for name, value in parameters.items():
        comment += " field_{}={:.16g}".format(name, value)
    output_lines = [str(len(combined)), comment]
    for bin_id, x, y, z, density, potential in combined:
        ix = index_maps[0][x]
        iy = index_maps[1][y]
        iz = index_maps[2][z]
        output_lines.append(
            "X {} {} {} {:.16g} {:.16g} {:.16g} {} {:.16g} {:.16g}".format(
                ix, iy, iz, x, y, z, bin_id, density, potential
            )
        )
    arguments.extxyz_output.write_text("\n".join(output_lines) + "\n")
    print(
        "Wrote {} grid points for {}; integrated density = {:.12g}".format(
            len(combined), arguments.application, density_integral
        )
    )


if __name__ == "__main__":
    main()
