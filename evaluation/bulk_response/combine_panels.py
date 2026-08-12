#!/usr/bin/env python3
"""Combine the four finalized bulk-response panels into one vector figure."""

import argparse
from pathlib import Path

import fitz


PANEL_FILES = (
    "panel_a_structure_factor.pdf",
    "panel_c_coexistence_curve.pdf",
    "panel_b_equation_of_state.pdf",
    "panel_d_interface_profiles.pdf",
)


def parse_arguments():
    script = Path(__file__).resolve()
    project = script.parents[2]
    default_results = project / "output" / "bulk_response"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-directory",
        type=Path,
        default=default_results,
        help="Directory containing the four panel PDF files.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=default_results,
    )
    parser.add_argument(
        "--png-dpi",
        type=float,
        default=500.0,
        help="Resolution of the preview PNG; the PDF remains vector-based.",
    )
    return parser.parse_args()


def panel_rectangle(column, row, panel_width, row_heights, column_gap, row_gap):
    """Return the destination rectangle for one panel in the 2-by-2 layout."""

    x0 = column * (panel_width + column_gap)
    y0 = row * (row_heights[0] + row_gap)
    return fitz.Rect(x0, y0, x0 + panel_width, y0 + row_heights[row])


def main():
    arguments = parse_arguments()
    panel_directory = arguments.panel_directory.expanduser().resolve()
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = [panel_directory / filename for filename in PANEL_FILES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing panel PDFs: {}".format(missing))

    sources = [fitz.open(path) for path in paths]
    try:
        panel_width = max(source[0].rect.width for source in sources)
        row_heights = (
            max(sources[index][0].rect.height for index in (0, 1)),
            max(sources[index][0].rect.height for index in (2, 3)),
        )
        column_gap = 4.0
        row_gap = 6.0
        page_width = 2.0 * panel_width + column_gap
        page_height = sum(row_heights) + row_gap

        combined = fitz.open()
        page = combined.new_page(width=page_width, height=page_height)
        for index, source in enumerate(sources):
            column = index % 2
            row = index // 2
            cell = panel_rectangle(
                column,
                row,
                panel_width,
                row_heights,
                column_gap,
                row_gap,
            )
            source_rect = source[0].rect
            # Preserve each panel's aspect ratio and align it with the top of
            # its row. Panel (d) is raised slightly within the lower row.
            scale = min(cell.width / source_rect.width, cell.height / source_rect.height)
            rendered_height = scale * source_rect.height
            y_offset = -4.0 if index == 3 else 0.0
            destination = fitz.Rect(
                cell.x0,
                cell.y0 + y_offset,
                cell.x0 + scale * source_rect.width,
                cell.y0 + y_offset + rendered_height,
            )
            page.show_pdf_page(destination, source, 0, keep_proportion=True)

        pdf_path = output_directory / "figure_bulk_response_to_phase_coexistence.pdf"
        png_path = output_directory / "figure_bulk_response_to_phase_coexistence.png"
        combined.set_metadata(
            {
                "title": "Bulk response to phase coexistence",
                "subject": "Structure factor, equation of state, coexistence, and interfaces",
            }
        )
        combined.save(pdf_path, garbage=4, deflate=True)
        matrix = fitz.Matrix(arguments.png_dpi / 72.0, arguments.png_dpi / 72.0)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(png_path)
        combined.close()
    finally:
        for source in sources:
            source.close()

    print("wrote", pdf_path)
    print("wrote", png_path)


if __name__ == "__main__":
    main()
