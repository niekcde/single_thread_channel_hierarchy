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
    rejected_peak_candidate_sigmas,
    score_peak_threshold_sigmas,
    heuristic_threshold_sigmas,
    terminal_threshold_sigmas,
    mode_sigmas,
    score,
    discarded_mode_candidate_sigmas=None,
):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    rejected_peak_candidate_sigmas = np.asarray(rejected_peak_candidate_sigmas, float)
    score_peak_threshold_sigmas = np.asarray(score_peak_threshold_sigmas, float)
    heuristic_threshold_sigmas = np.asarray(heuristic_threshold_sigmas, float)
    terminal_threshold_sigmas = np.asarray(terminal_threshold_sigmas, float)
    mode_sigmas = np.asarray(mode_sigmas, float)
    if discarded_mode_candidate_sigmas is None:
        discarded_mode_candidate_sigmas = np.array([], float)
    else:
        discarded_mode_candidate_sigmas = np.asarray(discarded_mode_candidate_sigmas, float)

    rejected_peak_style = dict(
        linestyle="-",
        linewidth=0.9,
        color="0.55",
        alpha=0.55,
    )
    score_peak_style = dict(
        linestyle="--",
        linewidth=1.1,
        color="tab:blue",
        alpha=0.9,
    )
    heuristic_threshold_style = dict(
        linestyle="-.",
        linewidth=1.2,
        color="tab:green",
        alpha=0.95,
    )
    terminal_threshold_style = dict(
        linestyle="--",
        linewidth=1.5,
        color="tab:red",
        alpha=0.95,
    )
    mode_style = dict(
        linestyle=":",
        linewidth=1.4,
        color="tab:orange",
        alpha=0.95,
    )
    discarded_mode_style = dict(
        linestyle="-",
        linewidth=1.0,
        color="tab:purple",
        alpha=0.65,
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
        for sigma in rejected_peak_candidate_sigmas:
            axis.axvline(sigma, **rejected_peak_style)
        for sigma in score_peak_threshold_sigmas:
            axis.axvline(sigma, **score_peak_style)
        for sigma in heuristic_threshold_sigmas:
            axis.axvline(sigma, **heuristic_threshold_style)
        for sigma in terminal_threshold_sigmas:
            axis.axvline(sigma, **terminal_threshold_style)
        for sigma in mode_sigmas:
            axis.axvline(sigma, **mode_style)
        for sigma in discarded_mode_candidate_sigmas:
            axis.axvline(sigma, **discarded_mode_style)

    ax[3].plot(sigmas[1:-1], score, linewidth=2)
    ax[3].set_xscale("log")
    ax[3].set_title("Boundary score")
    ax[3].set_xlabel("sigma (m)")
    ax[3].set_ylabel("score")

    legend_items = []
    if len(rejected_peak_candidate_sigmas):
        legend_items.append(
            Line2D([0], [0], label="rejected score-peak candidate", **rejected_peak_style)
        )
    if len(score_peak_threshold_sigmas):
        legend_items.append(
            Line2D([0], [0], label="score-peak threshold", **score_peak_style)
        )
    if len(heuristic_threshold_sigmas):
        legend_items.append(
            Line2D([0], [0], label="heuristic threshold", **heuristic_threshold_style)
        )
    if len(terminal_threshold_sigmas):
        legend_items.append(
            Line2D([0], [0], label="terminal threshold", **terminal_threshold_style)
        )
    if len(mode_sigmas):
        legend_items.append(Line2D([0], [0], label="final mode sigma", **mode_style))
    if len(discarded_mode_candidate_sigmas):
        legend_items.append(
            Line2D([0], [0], label="discarded mode candidate", **discarded_mode_style)
        )

    if legend_items:
        fig.legend(
            handles=legend_items,
            loc="upper center",
            ncol=min(4, len(legend_items)),
            frameon=False,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        plt.tight_layout()
    plt.show()


def plot_thresholding_from_result(result, use_threshold_modes: Optional[bool] = None):
    plot_data = result.get("plot_data", result)
    metrics = result.get("metrics", {})

    def pick(key, fallback=None):
        if key in plot_data:
            return plot_data[key]
        if key in result:
            return result[key]
        if key in metrics:
            return metrics[key]
        return fallback

    if use_threshold_modes is None:
        mode_sigmas = pick("mode_sigmas", [])
    elif use_threshold_modes:
        mode_sigmas = pick("threshold_mode_sigmas", pick("threshold_sigmas", []))
    else:
        mode_sigmas = pick("stable_mode_sigmas", pick("mode_sigmas", []))

    return plot_thresholding(
        sigmas=pick("sigmas"),
        sinu=pick("sinuosity"),
        turn=pick("turning"),
        dist=pick("dist"),
        rejected_peak_candidate_sigmas=pick("rejected_peak_candidate_sigmas", []),
        score_peak_threshold_sigmas=pick(
            "score_peak_threshold_sigmas",
            result.get("peak_boundary_sigmas", []),
        ),
        heuristic_threshold_sigmas=pick(
            "heuristic_threshold_sigmas",
            result.get("added_boundary_sigmas", []),
        ),
        terminal_threshold_sigmas=pick("terminal_threshold_sigmas", []),
        mode_sigmas=mode_sigmas,
        discarded_mode_candidate_sigmas=pick("discarded_mode_candidate_sigmas", []),
        score=pick("score"),
    )


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


def plot_modes_from_result(result, use_threshold_modes: Optional[bool] = None):
    if use_threshold_modes is None:
        modes = result.get("modes", [])
        mode_sigmas = result.get("mode_sigmas", [])
        mode_labels = result.get("mode_labels", [])
        sign_changes = result.get("curvature_sign_changes", [])
    elif use_threshold_modes:
        modes = result.get("threshold_modes", result.get("modes", []))
        mode_sigmas = result.get(
            "threshold_mode_sigmas",
            result.get("terminal_threshold_sigmas", result.get("mode_sigmas", [])),
        )
        mode_labels = result.get("threshold_mode_labels", result.get("mode_labels", []))
        sign_changes = result.get(
            "threshold_curvature_sign_changes",
            result.get("curvature_sign_changes", []),
        )
    else:
        modes = result.get("stable_modes", result.get("modes", []))
        mode_sigmas = result.get(
            "stable_mode_sigmas",
            result.get("mode_candidate_sigmas", result.get("mode_sigmas", [])),
        )
        mode_labels = result.get("stable_mode_labels", result.get("mode_labels", []))
        sign_changes = result.get(
            "stable_curvature_sign_changes",
            result.get("curvature_sign_changes", []),
        )

    if len(modes) == 0:
        return None

    labels = []

    for i, mode in enumerate(modes):
        if i < len(mode_labels):
            label = str(mode_labels[i])
        else:
            label = f"mode {i + 1}"
        if i < len(mode_sigmas):
            label = f"{label} | sigma~{float(mode_sigmas[i]):.0f}m"
        if i < len(sign_changes):
            label = f"{label} | sc={int(sign_changes[i])}"
        label = f"{label} | n={len(mode.coords)}"
        labels.append(label)

    return plot_modes(result["ls_equal"], modes, labels=labels)


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
