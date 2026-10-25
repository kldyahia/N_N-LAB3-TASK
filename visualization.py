"""Decision-boundary and sample visualization for a trained two-feature model."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiment import ExperimentResult


DISPLAY_NAMES = {
    "CulmenLength": "Culmen Length",
    "CulmenDepth": "Culmen Depth",
    "FlipperLength": "Flipper Length",
    "OriginLocation": "Origin Location (encoded category)",
    "BodyMass": "Body Mass",
}


def create_decision_boundary_figure(result: ExperimentResult, origin_mapping: dict[str, float]):
    """Create a 2-D plot of train/test data and the learned linear boundary."""
    # Construct a Figure directly (rather than pyplot) so report generation does not
    # spawn separate plot windows when called from the Tkinter application.
    fig = Figure(figsize=(8.5, 6.2), dpi=100)
    ax = fig.subplots()
    train_raw = result.scaler.impute(result.x_train_raw)
    test_raw = result.scaler.impute(result.x_test_raw)
    all_raw = np.vstack([train_raw, test_raw])

    x_min, x_max = float(np.min(all_raw[:, 0])), float(np.max(all_raw[:, 0]))
    y_min, y_max = float(np.min(all_raw[:, 1])), float(np.max(all_raw[:, 1]))
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = max(x_span * 0.08, 0.25 if result.feature_names[0] == "OriginLocation" else 0.5)
    y_pad = max(y_span * 0.08, 0.25 if result.feature_names[1] == "OriginLocation" else 0.5)
    gx, gy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 220),
        np.linspace(y_min - y_pad, y_max + y_pad, 220),
    )
    grid_raw = np.column_stack([gx.ravel(), gy.ravel()])
    grid_scaled = result.scaler.transform(grid_raw)
    regions = result.model.predict(grid_scaled).reshape(gx.shape)
    ax.contourf(gx, gy, regions, levels=[-1.5, 0, 1.5], alpha=0.10, cmap="coolwarm")
    scores = result.model.decision_function(grid_scaled).reshape(gx.shape)
    if float(np.min(scores)) <= 0.0 <= float(np.max(scores)):
        ax.contour(gx, gy, scores, levels=[0.0], colors=["#111827"], linewidths=2.0)

    styles = {
        (-1.0, "train"): {"color": "#2563eb", "marker": "o", "facecolors": "#2563eb"},
        (1.0, "train"): {"color": "#ea580c", "marker": "o", "facecolors": "#ea580c"},
        (-1.0, "test"): {"color": "#2563eb", "marker": "^", "facecolors": "none"},
        (1.0, "test"): {"color": "#ea580c", "marker": "^", "facecolors": "none"},
    }
    for label, class_name in ((-1.0, result.class_a), (1.0, result.class_b)):
        for split_name, points, labels in (
            ("train", train_raw, result.y_train),
            ("test", test_raw, result.y_test),
        ):
            subset = points[labels == label]
            style = styles[(label, split_name)]
            scatter_options = {
                "s": 45 if split_name == "train" else 62,
                "marker": style["marker"],
                "edgecolors": style["color"],
                "linewidths": 1.2,
                "alpha": 0.9,
                "zorder": 3,
            }
            if style["facecolors"] == "none":
                scatter_options["facecolors"] = "none"
            else:
                scatter_options["color"] = style["color"]
            ax.scatter(subset[:, 0], subset[:, 1], **scatter_options)

    feature_x, feature_y = result.feature_names
    ax.set_xlabel(DISPLAY_NAMES.get(feature_x, feature_x), fontsize=10)
    ax.set_ylabel(DISPLAY_NAMES.get(feature_y, feature_y), fontsize=10)
    ax.set_title(
        f"{result.algorithm}: {result.class_a} vs {result.class_b}\n"
        f"Features: {feature_x} + {feature_y} | Test accuracy: {result.accuracy:.1%}",
        fontsize=11,
        pad=12,
    )
    ax.grid(True, alpha=0.22, linestyle="--")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", label=f"{result.class_a} — train", markerfacecolor="#2563eb", markeredgecolor="#2563eb", markersize=7),
        Line2D([0], [0], marker="o", color="w", label=f"{result.class_b} — train", markerfacecolor="#ea580c", markeredgecolor="#ea580c", markersize=7),
        Line2D([0], [0], marker="^", color="w", label=f"{result.class_a} — test", markerfacecolor="none", markeredgecolor="#2563eb", markersize=7),
        Line2D([0], [0], marker="^", color="w", label=f"{result.class_b} — test", markerfacecolor="none", markeredgecolor="#ea580c", markersize=7),
        Line2D([0], [0], color="#111827", label="Decision boundary", linewidth=2),
    ]
    ax.legend(handles=legend_items, loc="best", fontsize=8, framealpha=0.92)

    if feature_x == "OriginLocation":
        ticks = sorted(set(origin_mapping.values()))
        labels = [next(name for name, code in origin_mapping.items() if code == tick) for tick in ticks]
        ax.set_xticks(ticks, labels, rotation=0)
    if feature_y == "OriginLocation":
        ticks = sorted(set(origin_mapping.values()))
        labels = [next(name for name, code in origin_mapping.items() if code == tick) for tick in ticks]
        ax.set_yticks(ticks, labels)

    fig.tight_layout()
    return fig
