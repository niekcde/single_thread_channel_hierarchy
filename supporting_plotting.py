"""Plotting helpers for inspecting extracted river line modes."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from shapely.geometry import LineString


def plot_thresholding(
    sigmas,
    sinu,
    turn,
    dist,
    peak_boundary_sigmas,
    added_boundary_sigmas,
    discarded_mode_candidate_sigmas,
    mode_sigmas,
    score,
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    peak_boundary_sigmas = np.asarray(peak_boundary_sigmas, float)
    added_boundary_sigmas = np.asarray(added_boundary_sigmas, float)
    discarded_mode_candidate_sigmas = np.asarray(discarded_mode_candidate_sigmas, float)
    mode_sigmas = np.asarray(mode_sigmas, float)

    peak_boundary_style = dict(
        linestyle="--",
        linewidth=1.1,
        color="tab:blue",
        alpha=0.9,
    )
    added_boundary_style = dict(
        linestyle="-.",
        linewidth=1.2,
        color="tab:green",
        alpha=0.95,
    )
    discarded_mode_style = dict(
        linestyle="-",
        linewidth=0.9,
        color="0.55",
        alpha=0.65,
    )
    mode_style = dict(
        linestyle=":",
        linewidth=1.4,
        color="tab:orange",
        alpha=0.95,
    )

    ax[0].plot(sigmas, sinu, linewidth=2)
    ax[0].set_xscale("log")
    ax[0].set_title("Sinuosity vs sigma")
    ax[0].set_xlabel("sigma (m)")
    ax[0].set_ylabel("sinuosity")

    ax[1].plot(sigmas, turn, linewidth=2)
    ax[1].set_xscale("log")
    ax[1].set_title("Turning energy vs sigma")
    ax[1].set_xlabel("sigma (m)")
    ax[1].set_ylabel("turning/length")

    ax[2].plot(sigmas, dist, linewidth=2)
    ax[2].set_xscale("log")
    ax[2].set_title("Mean deviation vs sigma")
    ax[2].set_xlabel("sigma (m)")
    ax[2].set_ylabel("mean distance (m)")

    for axis in ax:
        for sigma in peak_boundary_sigmas:
            axis.axvline(sigma, **peak_boundary_style)
        for sigma in added_boundary_sigmas:
            axis.axvline(sigma, **added_boundary_style)
        for sigma in discarded_mode_candidate_sigmas:
            axis.axvline(sigma, **discarded_mode_style)
        for sigma in mode_sigmas:
            axis.axvline(sigma, **mode_style)

    ax[3].plot(sigmas[1:-1], score, linewidth=2)
    ax[3].set_xscale("log")
    ax[3].set_title("Boundary score")
    ax[3].set_xlabel("sigma (m)")
    ax[3].set_ylabel("score")

    fig.legend(
        handles=[
            Line2D([0], [0], label="score-peak boundary sigma", **peak_boundary_style),
            Line2D([0], [0], label="added boundary sigma", **added_boundary_style),
            Line2D([0], [0], label="discarded mode candidate sigma", **discarded_mode_style),
            Line2D([0], [0], label="final mode sigma", **mode_style),
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def plot_modes(
    original: LineString,
    modes: Sequence[LineString],
    labels: Optional[Sequence[str]] = None,
):
    plt.figure(figsize=(10, 7))
    original_xy = np.asarray(original.coords, float)
    plt.plot(original_xy[:, 0], original_xy[:, 1], linewidth=2, label="original")

    for i, mode in enumerate(modes):
        xy = np.asarray(mode.coords, float)
        label = labels[i] if labels is not None else f"mode {i + 1}"
        plt.plot(xy[:, 0], xy[:, 1], linewidth=2, label=label)

    plt.axis("equal")
    plt.show()


def plot_modes_plotly(
    original: LineString,
    modes: Sequence[LineString],
    labels: Optional[Sequence[str]] = None,
    title: str = "Modes",
    show: bool = True,
):
    import plotly.graph_objects as go

    original_xy = np.asarray(original.coords, float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=original_xy[:, 0],
            y=original_xy[:, 1],
            mode="lines",
            name="original",
            line=dict(width=2),
        )
    )

    for i, mode in enumerate(modes):
        xy = np.asarray(mode.coords, float)
        label = labels[i] if labels is not None else f"mode {i + 1}"
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="lines",
                name=label,
                line=dict(width=4),
            )
        )

    fig.update_layout(
        title=title,
        width=1300,
        height=850,
        legend=dict(title=None),
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if show:
        fig.show()
    return fig


def _demo() -> None:
    x = np.linspace(0, 1000, 80)
    line = LineString(np.column_stack([x, 60 * np.sin(x / 100)]))
    mode = LineString(np.column_stack([x, 30 * np.sin(x / 180)]))
    plot_modes(line, [mode], labels=["demo mode"])


if __name__ == "__main__":
    _demo()
