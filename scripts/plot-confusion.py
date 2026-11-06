#!/usr/bin/env python3
"""Render the grounding diagnostic's saved confusion matrices without inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion(summary_path):
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    names = {"color_baseline": "Color baseline", "simple": "Simple prompt", "columns": "Explicit-column prompt"}
    labels = ["left", "center", "right"]
    predictions = labels + ["invalid"]
    matrices = [np.array([[summary["results"][name]["confusion"][truth][pred] for pred in predictions] for truth in labels]) for name in names]
    vmax = max(int(matrix.max()) for matrix in matrices)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for ax, (name, title), matrix in zip(axes, names.items(), matrices):
        heatmap = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(vmax, 1))
        for row in range(3):
            for col in range(4):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > vmax/2 else "black", fontsize=13)
        result = summary["results"][name]
        ax.set_title(f"{title}\nAccuracy: {result['accuracy']:.1%}")
        ax.set_xticks(range(4), predictions)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("Predicted direction")
        ax.set_ylabel("True direction")
    fig.colorbar(heatmap, ax=axes, label="Number of questions", shrink=0.65)
    fig.suptitle(f"Validation grounding diagnostic — {summary['scenes']} scenes\nRows = ground truth; columns = prediction; center = middle", fontsize=14)
    out = summary_path.with_name("confusion_matrices.png")
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    plot_confusion(parser.parse_args().summary)
