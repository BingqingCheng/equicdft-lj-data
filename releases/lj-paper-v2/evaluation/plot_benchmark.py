"""Redraw the selected benchmark from its retained display cache and full metrics."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data/study/cDFT"
sys.path.insert(0, str(STUDY / "benchmark-plot-scripts"))
import combined_benchmark_2x2 as style


def main():
    source = ROOT / "source_data/fig06_benchmarks"
    metadata = json.loads((source / "provenance_and_metrics.json").read_text())
    arrays = np.load(source / "figure_points.npz")
    style.configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(7, 7))
    figure.subplots_adjust(left=.105, right=.985, bottom=.09, top=.945, wspace=.29, hspace=.30)
    setups = [
        ("forward", r"Reference $\beta V^{\mathrm{ext}}$", r"Predicted $\beta V^{\mathrm{ext}}$", r"NVT: $\rho\rightarrow V^{\mathrm{ext}}$", 3.5, .13, (.002, .998)),
        ("solve", r"Reference $\rho\sigma^3$", r"Solved $\rho\sigma^3$", r"NVT: $V^{\mathrm{ext}}\rightarrow\rho$", 3.5, .13, (0, .997)),
        ("large", r"Reference $\rho\sigma^3$", r"Solved $\rho\sigma^3$", r"NVT: small$\rightarrow$large", 3.5, .13, (0, .9995)),
        ("gcmc", r"Reference $\Delta(\beta\mu)$", r"Predicted $\Delta(\beta\mu)$", r"NVT$\rightarrow$GCMC", 16, .65, (0, 1))]
    for axis, label, setup in zip(axes.ravel(), "abcd", setups):
        task, xlabel, ylabel, title, size, alpha, quantiles = setup
        group = {float(T): {**row, "display": (arrays[f"{task}_T{T.replace('.', 'p')}_target"],
                    arrays[f"{task}_T{T.replace('.', 'p')}_prediction"])}
                 for T, row in metadata["metrics"][task].items()}
        temperatures = style.GCMC_TEMPERATURES if task == "gcmc" else style.HELD_OUT_TEMPERATURES
        limits = style.square_limits({T: row["display"] for T, row in group.items()}, *quantiles)
        if task in ("solve", "large"):
            limits = (max(0, limits[0]), limits[1])
        style.draw_panel(axis, group, temperatures, xlabel, ylabel, limits, title, size, alpha, 3)
        style.add_panel_label(axis, label)
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    figure.savefig(output / "combined_benchmark_C4.pdf")
    figure.savefig(output / "combined_benchmark_C4.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
