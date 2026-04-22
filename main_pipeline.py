"""Main river hierarchy line-mode extraction pipeline.

Notebook usage:

    from main_pipeline import extract_line_modes_auto

Command-line smoke demo:

    python main_pipeline.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString

from supporting_boundaries import (
    detect_mode_boundaries_autoK,
    maybe_add_terminal_mode,
    pick_axis_sigma,
    prune_redundant_adjacent_modes,
    representative_modes_by_stability,
    try_insert_mid_boundary,
)
from supporting_geometry import (
    resample_linestring_equal,
    simplify_mode,
    smooth_linestring_gaussian,
    snap_vertices_to_original,
)
from supporting_metrics import (
    classify_modes_by_extrema,
    global_sinuosity,
    mean_distance_to_original,
    turning_energy,
)


def choose_step_m(ls: LineString, width_m=None, min_step: float = 2.0) -> float:
    length = float(ls.length)
    if width_m is not None:
        width = float(width_m)
        max_step = max(6.0, min(15.0, width / 1.5))
        return float(np.clip(width / 4.0, min_step, max_step))

    return float(np.clip(length / 8000.0, min_step, 12.0))


def choose_sigmas(
    ls_eq: LineString,
    step_m: float,
    width_m=None,
    n_sigmas: int = 120,
):
    sigma_min = 2.0 * float(step_m)
    length = float(ls_eq.length)

    if width_m is not None:
        width = float(width_m)
        sigma_max = max(length / 3.0, 80.0 * width)
    else:
        sigma_max = max(length / 3.0, 8000.0)

    return np.geomspace(sigma_min, sigma_max, int(n_sigmas))


def build_scale_space_smooth_only(
    ls_eq: LineString,
    sigmas_m,
    step_m: float,
):
    reps = []
    sinuosity = []
    turning = []
    dist = []

    for sigma in sigmas_m:
        smoothed = smooth_linestring_gaussian(ls_eq, float(sigma), float(step_m))
        reps.append(smoothed)
        sinuosity.append(global_sinuosity(smoothed))
        turning.append(turning_energy(smoothed))
        dist.append(mean_distance_to_original(ls_eq, smoothed, n=600))

    metrics = {
        "sinuosity": np.asarray(sinuosity, float),
        "turning": np.asarray(turning, float),
        "dist": np.asarray(dist, float),
    }
    return reps, metrics


def extract_line_modes_auto(
    ls: LineString,
    *,
    width_m=None,
    step_m=None,
    n_sigmas: int = 120,
    min_sep: int = 10,
    smooth_w: int = 7,
    score_percentile: float = 80,
    prominence_percentile: float = 70,
    left_right: int = 12,
    use_relative_jumps: bool = True,
    sinu_jump_min: float = 0.05,
    dist_jump_min=None,
    turn_jump_min: float = 0.00015,
    sinu_jump_rel: float = 0.02,
    dist_jump_rel: float = 0.03,
    turn_jump_rel: float = 0.05,
    max_boundaries: int = 6,
    gap_ratio_stop: float = 1.8,
    eps_floor=None,
    eps_factor: float = 1.0,
    eps_power: float = 0.5,
    snap_to_original: bool = False,
    prune_dist_abs: float = 30.0,
    prune_dist_w_frac: float = 0.30,
    prune_sinu_abs: float = 0.04,
    prune_sc_drop_frac: float = 0.20,
    allow_mid_insertion: bool = True,
    allow_terminal: bool = True,
    terminal_tail_frac: float = 0.20,
    terminal_min_prom_frac: float = 0.35,
    terminal_min_sinu_drop: float = 0.03,
    terminal_min_dist_increase: float = 0.10,
    make_plots: bool = True,
) -> Dict[str, Any]:
    if step_m is None:
        step_m = choose_step_m(ls, width_m=width_m)
    if eps_floor is None:
        eps_floor = float(step_m)

    ls_eq = resample_linestring_equal(ls, step_m)
    sigmas = choose_sigmas(ls_eq, step_m=step_m, width_m=width_m, n_sigmas=n_sigmas)

    reps, metrics = build_scale_space_smooth_only(ls_eq, sigmas, step_m)
    sinuosity = metrics["sinuosity"]
    turning = metrics["turning"]
    dist = metrics["dist"]

    if dist_jump_min is None:
        if width_m is not None:
            dist_jump_min = 1.5 * float(width_m)
        else:
            dist_jump_min = float(ls_eq.length) / 200.0

    idx, _, score = detect_mode_boundaries_autoK(
        sigmas,
        sinuosity,
        turning,
        dist,
        min_sep=min_sep,
        smooth_w=smooth_w,
        score_percentile=score_percentile,
        prominence_percentile=prominence_percentile,
        left_right=left_right,
        use_relative_jumps=use_relative_jumps,
        sinu_jump_min=sinu_jump_min,
        dist_jump_min=float(dist_jump_min),
        turn_jump_min=turn_jump_min,
        sinu_jump_rel=sinu_jump_rel,
        dist_jump_rel=dist_jump_rel,
        turn_jump_rel=turn_jump_rel,
        max_boundaries=max_boundaries,
        gap_ratio_stop=gap_ratio_stop,
    )

    axis_fallback_used = False
    if len(idx) == 0:
        axis_fallback_used = True
        axis_sigma = pick_axis_sigma(sigmas, sinuosity, turning, dist)
        axis_idx = int(np.argmin(np.abs(np.log(sigmas) - np.log(axis_sigma))))
        idx = [axis_idx]

    def build_and_prune(idx_list: List[int]) -> Tuple[np.ndarray, List[LineString]]:
        mode_sigmas, mode_lines_smooth = representative_modes_by_stability(
            sigmas, reps, idx_list, score
        )

        modes = []
        for sigma, mode in zip(mode_sigmas, mode_lines_smooth):
            out = simplify_mode(
                mode,
                sigma=float(sigma),
                eps_floor=float(eps_floor),
                eps_factor=float(eps_factor),
                power=float(eps_power),
            )
            if snap_to_original:
                out = snap_vertices_to_original(out, ls_eq)
            modes.append(out)

        modes, mode_sigmas = prune_redundant_adjacent_modes(
            modes,
            mode_sigmas,
            original=ls_eq,
            width_m=width_m,
            dist_abs=prune_dist_abs,
            first_dist_w_mult=3.0,
            adj_dist_w_mult=3.0,
            dist_w_frac=prune_dist_w_frac,
            turn_frac_min=0.15,
            sc_frac_min=0.25,
        )

        return mode_sigmas, modes

    idx = sorted(set(idx))
    mode_sigmas, modes = build_and_prune(idx)

    mid_info = {"mid_added": False, "reason": "disabled"}
    if allow_mid_insertion:
        idx2, mid_info = try_insert_mid_boundary(
            idx,
            sigmas=sigmas,
            reps=reps,
            score=score,
            ls_eq=ls_eq,
            width_m=width_m,
            min_sep=min_sep,
            eps_floor=eps_floor,
            eps_factor=eps_factor,
            eps_power=eps_power,
            snap_to_original=snap_to_original,
        )
        if idx2 != idx:
            idx = idx2
            mode_sigmas, modes = build_and_prune(idx)

    terminal_info = {"added_terminal": False, "reason": "disabled"}
    if allow_terminal:
        mode_sigmas, modes, terminal_info = maybe_add_terminal_mode(
            sigmas,
            reps,
            modes,
            mode_sigmas,
            score,
            tail_frac=terminal_tail_frac,
            min_prom_frac=terminal_min_prom_frac,
            min_sinu_drop=terminal_min_sinu_drop,
            min_dist_increase=terminal_min_dist_increase,
            ls_eq=ls_eq,
        )

    if len(modes):
        sign_changes, lobes, labels = classify_modes_by_extrema(modes)
    else:
        sign_changes, lobes, labels = np.array([]), np.array([]), []

    boundary_sigmas_final = (
        sigmas[np.asarray(sorted(set(idx)), dtype=int)] if len(idx) else np.array([])
    )

    if make_plots:
        from supporting_plotting import plot_modes, plot_thresholding

        plot_thresholding(
            sigmas,
            sinuosity,
            turning,
            dist,
            boundary_sigmas_final,
            score,
            terminal_info,
        )

        plot_labels = []
        for sigma, mode, label, sign_change in zip(
            mode_sigmas, modes, labels, sign_changes
        ):
            plot_labels.append(
                f"{label} | sigma~{sigma:.0f}m | sc={sign_change} | n={len(mode.coords)}"
            )
        plot_modes(ls_eq, modes, labels=plot_labels)

    return {
        "ls_equal": ls_eq,
        "step_m": float(step_m),
        "width_m": None if width_m is None else float(width_m),
        "sigmas": sigmas,
        "metrics": metrics,
        "score": score,
        "boundary_indices": sorted(set(idx)),
        "boundary_sigmas": boundary_sigmas_final,
        "axis_fallback_used": axis_fallback_used,
        "mid_info": mid_info,
        "terminal_info": terminal_info,
        "mode_sigmas": mode_sigmas,
        "modes": modes,
        "curvature_sign_changes": sign_changes,
        "curvature_lobes": lobes,
        "mode_labels": labels,
    }


def _demo() -> None:
    x = np.linspace(0, 8000, 500)
    y = 300 * np.sin(x / 250) + 900 * np.sin(x / 1800)
    line = LineString(np.column_stack([x, y]))

    result = extract_line_modes_auto(
        line,
        width_m=120,
        n_sigmas=45,
        make_plots=False,
        allow_terminal=False,
    )

    print("step_m:", result["step_m"])
    print("boundary sigmas:", np.round(result["boundary_sigmas"], 2).tolist())
    print("mode sigmas:", np.round(result["mode_sigmas"], 2).tolist())
    print("labels:", result["mode_labels"])


if __name__ == "__main__":
    _demo()
