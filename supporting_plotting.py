"""Plotting helpers for inspecting extracted river line modes."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString


def plot_thresholding(sigmas, sinu, turn, dist, boundary_sigmas, score, terminal_info):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))

    if terminal_info.get("added_terminal") is True:
        ax[3].axvline(sigmas[-1], linestyle="--", linewidth=1)

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

    for axis in ax[:3]:
        for sigma in boundary_sigmas:
            axis.axvline(sigma, linestyle="--", linewidth=1)

    ax[3].plot(sigmas[1:-1], score, linewidth=2)
    ax[3].set_xscale("log")
    ax[3].set_title("Boundary score")
    ax[3].set_xlabel("sigma (m)")
    ax[3].set_ylabel("score")
    for sigma in boundary_sigmas:
        ax[3].axvline(sigma, linestyle="--", linewidth=1)

    plt.tight_layout()
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
