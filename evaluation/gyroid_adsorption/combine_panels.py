#!/usr/bin/env python3
"""Stack the finished gyroid density and adsorption PDFs without redrawing."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation


def read_single_page(path: Path):
    """Read one finished panel PDF and require exactly one page."""

    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError("{} must contain exactly one page".format(path))
    return reader, reader.pages[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_directory",
        type=Path,
        help="model applications-soft directory containing both finished PDFs",
    )
    parser.add_argument("--density", type=Path)
    parser.add_argument("--adsorption", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gap-points",
        type=float,
        default=6.0,
        help="vertical white gap between the two source pages in PDF points",
    )
    arguments = parser.parse_args()

    results = arguments.results_directory.expanduser().resolve()
    density_path = arguments.density
    if density_path is None:
        density_path = results / "gyroid_density_multicontour.pdf"
    density_path = density_path.expanduser().resolve()

    adsorption_path = arguments.adsorption
    if adsorption_path is None:
        adsorption_path = results / "gyroid_adsorption_two_panel_T1.2.pdf"
    adsorption_path = adsorption_path.expanduser().resolve()

    output = arguments.output
    if output is None:
        output = results / "gyroid_application_combined_T1.2.pdf"
    output = output.expanduser().resolve()

    density_reader, density_page = read_single_page(density_path)
    adsorption_reader, adsorption_page = read_single_page(adsorption_path)
    density_width = float(density_page.mediabox.width)
    adsorption_width = float(adsorption_page.mediabox.width)
    if abs(density_width - adsorption_width) > 0.1:
        raise ValueError(
            "source PDF widths differ: {:.3f} versus {:.3f} points".format(
                density_width,
                adsorption_width,
            )
        )
    if arguments.gap_points < 0.0:
        raise ValueError("--gap-points must be non-negative")

    density_height = float(density_page.mediabox.height)
    adsorption_height = float(adsorption_page.mediabox.height)
    combined_height = density_height + adsorption_height + arguments.gap_points

    writer = PdfWriter()
    combined = writer.add_blank_page(
        width=density_width,
        height=combined_height,
    )
    combined.merge_transformed_page(
        adsorption_page,
        Transformation().translate(tx=0.0, ty=0.0),
    )
    combined.merge_transformed_page(
        density_page,
        Transformation().translate(
            tx=0.0,
            ty=adsorption_height + arguments.gap_points,
        ),
    )

    # Keep the source readers alive until writing completes: their page
    # resources are referenced while pypdf constructs the combined page.
    _ = density_reader, adsorption_reader
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)
    print("wrote {}".format(output))


if __name__ == "__main__":
    main()
