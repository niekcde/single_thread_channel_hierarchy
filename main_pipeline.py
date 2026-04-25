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
    detect_boundary_thresholds_autoK,
    maybe_add_terminal_threshold,
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
    resolve_distance_sampling,
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
    sigma_min: float | None = None,
    sigma_max: float | None = None,
):
    if sigma_min is None:
        sigma_min = 2.0 * float(step_m)
    else:
        sigma_min = float(sigma_min)
    length = float(ls_eq.length)

    if sigma_max is None:
        if width_m is not None:
            width = float(width_m)
            sigma_max = max(length / 3.0, 80.0 * width)
        else:
            sigma_max = max(length / 3.0, 8000.0)
    else:
        sigma_max = float(sigma_max)

    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min")

    return np.geomspace(sigma_min, sigma_max, int(n_sigmas))


def build_scale_space_smooth_only(
    ls_eq: LineString,
    sigmas_m,
    step_m: float,
    *,
    dist_sample_spacing: float | None = None,
    dist_min_samples: int = 200,
    dist_max_samples: int = 4000,
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
        dist.append(
            mean_distance_to_original(
                ls_eq,
                smoothed,
                sample_spacing=dist_sample_spacing,
                min_samples=dist_min_samples,
                max_samples=dist_max_samples,
            )
        )

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
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    peak_selection_mode: str = "adaptive",
    use_score_percentile_gate: bool = False,
    min_sep: int = 10,
    smooth_w: int = 7,
    score_percentile: float = 80,
    prominence_percentile: float = 70,
    prominence_mad_mult: float = 1.5,
    contrast_low: float = 0.15,
    contrast_high: float = 0.30,
    borderline_prominence_mult: float = 1.25,
    left_right: int = 12,
    use_relative_jumps: bool = True,
    sinu_jump_min: float = 0.05,
    dist_jump_min=None,
    turn_jump_min: float = 0.00015,
    sinu_jump_rel: float = 0.02,
    dist_jump_rel: float = 0.03,
    turn_jump_rel: float = 0.05,
    dist_sample_spacing: float | None = None,
    dist_sample_min_samples: int = 200,
    dist_sample_max_samples: int = 4000,
    max_boundaries: int = 6,
    gap_ratio_stop: float = 1.8,
    eps_floor=None,
    eps_factor: float = 1.0,
    eps_power: float = 0.5,
    snap_to_original: bool = False,
    derive_modes: bool = True,
    prune_modes: bool = False,
    prune_dist_abs: float = 30.0,
    prune_dist_w_frac: float = 0.30,
    prune_sinu_abs: float = 0.04,
    prune_sc_drop_frac: float = 0.20,
    allow_axis_fallback: bool = False,
    allow_mid_insertion: bool = False,
    allow_terminal: bool = True,
    terminal_tail_frac: float = 0.20,
    terminal_min_prom_frac: float = 0.35,
    terminal_min_sinu_drop: float = 0.03,
    terminal_min_dist_increase: float = 0.10,
    use_threshold_sigmas_as_modes: bool = False,
    make_plots: bool = True,
) -> Dict[str, Any]:
    if step_m is None:
        step_m = choose_step_m(ls, width_m=width_m)
    if eps_floor is None:
        eps_floor = float(step_m)
    if dist_sample_spacing is None:
        dist_sample_spacing = float(max(20.0 * step_m, 50.0))

    ls_eq = resample_linestring_equal(ls, step_m)
    distance_sampling = resolve_distance_sampling(
        ls_eq,
        sample_spacing=dist_sample_spacing,
        min_samples=dist_sample_min_samples,
        max_samples=dist_sample_max_samples,
    )
    sigmas = choose_sigmas(
        ls_eq,
        step_m=step_m,
        width_m=width_m,
        n_sigmas=n_sigmas,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    reps, metrics = build_scale_space_smooth_only(
        ls_eq,
        sigmas,
        step_m,
        dist_sample_spacing=dist_sample_spacing,
        dist_min_samples=dist_sample_min_samples,
        dist_max_samples=dist_sample_max_samples,
    )
    sinuosity = metrics["sinuosity"]
    turning = metrics["turning"]
    dist = metrics["dist"]

    if dist_jump_min is None:
        if width_m is not None:
            dist_jump_min = 1.5 * float(width_m)
        else:
            dist_jump_min = float(ls_eq.length) / 200.0

    peak_threshold_indices, _, score, boundary_diagnostics = detect_boundary_thresholds_autoK(
        sigmas,
        sinuosity,
        turning,
        dist,
        selection_mode=peak_selection_mode,
        use_score_percentile_gate=use_score_percentile_gate,
        min_sep=min_sep,
        smooth_w=smooth_w,
        score_percentile=score_percentile,
        prominence_percentile=prominence_percentile,
        prominence_mad_mult=prominence_mad_mult,
        contrast_low=contrast_low,
        contrast_high=contrast_high,
        borderline_prominence_mult=borderline_prominence_mult,
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
        return_diagnostics=True,
    )
    peak_threshold_indices = sorted(set(int(k) for k in peak_threshold_indices))
    final_threshold_indices = list(peak_threshold_indices)
    heuristic_threshold_indices: List[int] = []
    terminal_threshold_indices: List[int] = []

    axis_info = {
        "axis_fallback_used": False,
        "reason": "disabled" if not allow_axis_fallback else "not needed",
    }
    if len(final_threshold_indices) == 0 and allow_axis_fallback:
        axis_sigma = pick_axis_sigma(sigmas, sinuosity, turning, dist)
        axis_idx = int(np.argmin(np.abs(np.log(sigmas) - np.log(axis_sigma))))
        final_threshold_indices = [axis_idx]
        heuristic_threshold_indices = [axis_idx]
        axis_info = {
            "axis_fallback_used": True,
            "reason": "accepted",
            "sigma_idx": int(axis_idx),
            "sigma": float(sigmas[axis_idx]),
        }

    mid_info = {"mid_added": False, "reason": "disabled"}
    if allow_mid_insertion and len(final_threshold_indices) > 0:
        idx2, mid_info = try_insert_mid_boundary(
            final_threshold_indices,
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
            dist_abs=prune_dist_abs,
            dist_w_frac=prune_dist_w_frac,
            turn_frac_min=0.15,
            sc_frac_min=prune_sc_drop_frac,
            sinu_abs_min=prune_sinu_abs,
            dist_sample_spacing=dist_sample_spacing,
            dist_min_samples=dist_sample_min_samples,
            dist_max_samples=dist_sample_max_samples,
        )
        if idx2 != final_threshold_indices:
            heuristic_threshold_indices.extend(
                [int(k) for k in idx2 if int(k) not in final_threshold_indices]
            )
            final_threshold_indices = sorted(set(int(k) for k in idx2))
    elif allow_mid_insertion:
        mid_info = {"mid_added": False, "reason": "requires existing thresholds"}

    terminal_info = {
        "added_terminal_threshold": False,
        "reason": "disabled" if not allow_terminal else "not added",
    }
    if allow_terminal:
        idx2, terminal_info = maybe_add_terminal_threshold(
            sigmas,
            score,
            final_threshold_indices,
            sinuosity=sinuosity,
            turning=turning,
            dist=dist,
            tail_frac=terminal_tail_frac,
            min_prom_frac=terminal_min_prom_frac,
            min_sep=min_sep,
            left_right=left_right,
            score_percentile=score_percentile,
            use_relative_jumps=use_relative_jumps,
            sinu_jump_min=sinu_jump_min,
            dist_jump_min=float(dist_jump_min),
            turn_jump_min=turn_jump_min,
            sinu_jump_rel=sinu_jump_rel,
            dist_jump_rel=dist_jump_rel,
            turn_jump_rel=turn_jump_rel,
            min_sinu_drop=terminal_min_sinu_drop,
            min_dist_increase=terminal_min_dist_increase,
        )
        if idx2 != final_threshold_indices:
            terminal_threshold_indices = [
                int(k) for k in idx2 if int(k) not in final_threshold_indices
            ]
            final_threshold_indices = sorted(set(int(k) for k in idx2))
    terminal_info["added_terminal"] = bool(
        terminal_info.get("added_terminal_threshold", False)
    )

    final_threshold_indices = sorted(set(int(k) for k in final_threshold_indices))
    threshold_sigmas = (
        sigmas[np.asarray(final_threshold_indices, dtype=int)]
        if len(final_threshold_indices)
        else np.array([])
    )
    score_peak_threshold_sigmas = (
        sigmas[np.asarray(peak_threshold_indices, dtype=int)]
        if len(peak_threshold_indices)
        else np.array([])
    )
    heuristic_threshold_indices = sorted(set(int(k) for k in heuristic_threshold_indices))
    heuristic_threshold_sigmas = (
        sigmas[np.asarray(heuristic_threshold_indices, dtype=int)]
        if len(heuristic_threshold_indices)
        else np.array([])
    )
    terminal_threshold_sigmas = (
        sigmas[np.asarray(terminal_threshold_indices, dtype=int)]
        if len(terminal_threshold_indices)
        else np.array([])
    )
    score_peak_candidate_indices = [
        int(record["sigma_idx"]) for record in boundary_diagnostics.get("candidates", [])
    ]
    rejected_peak_candidate_indices = [
        int(record["sigma_idx"])
        for record in boundary_diagnostics.get("candidates", [])
        if record.get("decision") != "kept"
    ]
    score_peak_candidate_sigmas = (
        sigmas[np.asarray(score_peak_candidate_indices, dtype=int)]
        if len(score_peak_candidate_indices)
        else np.array([])
    )
    rejected_peak_candidate_sigmas = (
        sigmas[np.asarray(sorted(set(rejected_peak_candidate_indices)), dtype=int)]
        if len(rejected_peak_candidate_indices)
        else np.array([])
    )

    stable_mode_candidate_sigmas = np.array([])
    stable_mode_sigmas = np.array([])
    stable_modes: List[LineString] = []
    stable_mode_sign_changes = np.array([])
    stable_mode_lobes = np.array([])
    stable_mode_labels: List[str] = []
    discarded_mode_candidate_sigmas = np.array([])
    threshold_mode_sigmas = np.array([])
    threshold_modes: List[LineString] = []
    threshold_mode_sign_changes = np.array([])
    threshold_mode_lobes = np.array([])
    threshold_mode_labels: List[str] = []
    mode_sigmas = np.array([])
    modes: List[LineString] = []
    sign_changes = np.array([])
    lobes = np.array([])
    labels: List[str] = []
    prune_diagnostics: Dict[str, Any] = {
        "enabled": False,
        "reason": "mode pruning disabled" if derive_modes else "mode derivation disabled",
        "candidates": [],
        "fallback": {"used": False},
    }

    def materialize_modes(candidate_sigmas, candidate_lines):
        out_modes = []
        for sigma, mode in zip(np.asarray(candidate_sigmas, float), candidate_lines):
            out = simplify_mode(
                mode,
                sigma=float(sigma),
                eps_floor=float(eps_floor),
                eps_factor=float(eps_factor),
                power=float(eps_power),
            )
            if snap_to_original:
                out = snap_vertices_to_original(out, ls_eq)
            out_modes.append(out)
        return out_modes

    if derive_modes:
        stable_mode_candidate_sigmas, stable_mode_lines_smooth = representative_modes_by_stability(
            sigmas, reps, final_threshold_indices, score
        )
        stable_mode_lines = materialize_modes(
            stable_mode_candidate_sigmas,
            stable_mode_lines_smooth,
        )

        if prune_modes:
            stable_modes, stable_mode_sigmas, prune_diagnostics = prune_redundant_adjacent_modes(
                stable_mode_lines,
                stable_mode_candidate_sigmas,
                original=ls_eq,
                width_m=width_m,
                dist_abs=prune_dist_abs,
                first_dist_w_mult=3.0,
                adj_dist_w_mult=3.0,
                dist_w_frac=prune_dist_w_frac,
                turn_frac_min=0.15,
                sc_frac_min=prune_sc_drop_frac,
                sinu_abs_min=prune_sinu_abs,
                dist_sample_spacing=dist_sample_spacing,
                dist_min_samples=dist_sample_min_samples,
                dist_max_samples=dist_sample_max_samples,
                return_diagnostics=True,
            )
            stable_mode_sigmas = np.asarray(stable_mode_sigmas, float)
            discarded_mode_candidate_sigmas = np.array(
                [
                    sigma
                    for sigma in np.asarray(stable_mode_candidate_sigmas, float)
                    if not np.any(np.isclose(stable_mode_sigmas, sigma))
                ],
                dtype=float,
            )
        else:
            stable_modes = stable_mode_lines
            stable_mode_sigmas = np.asarray(stable_mode_candidate_sigmas, float)

        if len(stable_modes):
            stable_mode_sign_changes, stable_mode_lobes, stable_mode_labels = classify_modes_by_extrema(
                stable_modes
            )

        if len(final_threshold_indices):
            threshold_mode_sigmas = sigmas[np.asarray(final_threshold_indices, dtype=int)]
            threshold_mode_lines_smooth = [reps[int(k)] for k in final_threshold_indices]
            threshold_modes = materialize_modes(
                threshold_mode_sigmas,
                threshold_mode_lines_smooth,
            )
            if len(threshold_modes):
                (
                    threshold_mode_sign_changes,
                    threshold_mode_lobes,
                    threshold_mode_labels,
                ) = classify_modes_by_extrema(threshold_modes)

        if use_threshold_sigmas_as_modes:
            modes = threshold_modes
            mode_sigmas = np.asarray(threshold_mode_sigmas, float)
            sign_changes = np.asarray(threshold_mode_sign_changes)
            lobes = np.asarray(threshold_mode_lobes)
            labels = list(threshold_mode_labels)
        else:
            modes = stable_modes
            mode_sigmas = np.asarray(stable_mode_sigmas, float)
            sign_changes = np.asarray(stable_mode_sign_changes)
            lobes = np.asarray(stable_mode_lobes)
            labels = list(stable_mode_labels)

    mode_sigma_source = "threshold" if use_threshold_sigmas_as_modes else "stability"

    plot_data = {
        "sigmas": sigmas,
        "sinuosity": sinuosity,
        "turning": turning,
        "dist": dist,
        "score": score,
        "threshold_sigmas": threshold_sigmas,
        "score_peak_candidate_sigmas": score_peak_candidate_sigmas,
        "rejected_peak_candidate_sigmas": rejected_peak_candidate_sigmas,
        "score_peak_threshold_sigmas": score_peak_threshold_sigmas,
        "heuristic_threshold_sigmas": heuristic_threshold_sigmas,
        "terminal_threshold_sigmas": terminal_threshold_sigmas,
        "stable_mode_sigmas": stable_mode_sigmas,
        "threshold_mode_sigmas": threshold_mode_sigmas,
        "mode_sigmas": mode_sigmas,
        "discarded_mode_candidate_sigmas": discarded_mode_candidate_sigmas,
        "mode_sigma_source": mode_sigma_source,
    }

    boundary_diagnostics["threshold_indices"] = [int(k) for k in final_threshold_indices]
    boundary_diagnostics["threshold_sigmas"] = [float(s) for s in threshold_sigmas]
    boundary_diagnostics["score_peak_threshold_indices"] = [
        int(k) for k in peak_threshold_indices
    ]
    boundary_diagnostics["score_peak_threshold_sigmas"] = [
        float(s) for s in score_peak_threshold_sigmas
    ]
    boundary_diagnostics["heuristic_threshold_indices"] = [
        int(k) for k in heuristic_threshold_indices
    ]
    boundary_diagnostics["heuristic_threshold_sigmas"] = [
        float(s) for s in heuristic_threshold_sigmas
    ]
    boundary_diagnostics["terminal_threshold_indices"] = [
        int(k) for k in terminal_threshold_indices
    ]
    boundary_diagnostics["terminal_threshold_sigmas"] = [
        float(s) for s in terminal_threshold_sigmas
    ]
    boundary_diagnostics["score_peak_candidate_indices"] = [
        int(k) for k in score_peak_candidate_indices
    ]
    boundary_diagnostics["score_peak_candidate_sigmas"] = [
        float(s) for s in score_peak_candidate_sigmas
    ]
    boundary_diagnostics["rejected_peak_candidate_indices"] = [
        int(k) for k in rejected_peak_candidate_indices
    ]
    boundary_diagnostics["rejected_peak_candidate_sigmas"] = [
        float(s) for s in rejected_peak_candidate_sigmas
    ]

    near_original_mult = 3.0
    near_original_threshold = (
        near_original_mult * float(width_m) if width_m is not None else float(prune_dist_abs)
    )
    near_original_candidate_indices = []
    near_original_threshold_indices = []
    boundary_diagnostics["near_original_flag_settings"] = {
        "enabled": True,
        "comparison_basis": "original",
        "distance_threshold": float(near_original_threshold),
        "distance_threshold_rule": (
            f"{near_original_mult:.1f} * width_m" if width_m is not None else "prune_dist_abs"
        ),
        "distance_multiplier": float(near_original_mult),
        "width_m": None if width_m is None else float(width_m),
        "dist_sample_spacing": None
        if dist_sample_spacing is None
        else float(dist_sample_spacing),
        "dist_min_samples": int(dist_sample_min_samples),
        "dist_max_samples": int(dist_sample_max_samples),
    }
    for record in boundary_diagnostics.get("candidates", []):
        k = int(record["sigma_idx"])
        distance_to_original, sampling = mean_distance_to_original(
            ls_eq,
            reps[k],
            sample_spacing=dist_sample_spacing,
            min_samples=dist_sample_min_samples,
            max_samples=dist_sample_max_samples,
            return_diagnostics=True,
        )
        near_original_flag = bool(distance_to_original < near_original_threshold)
        record["distance_to_original"] = float(distance_to_original)
        record["near_original_threshold"] = float(near_original_threshold)
        record["near_original_ratio"] = float(
            distance_to_original / max(near_original_threshold, 1e-12)
        )
        record["near_original_flag"] = near_original_flag
        record["near_original_sampling"] = sampling
        if width_m is not None:
            if distance_to_original < float(width_m):
                record["near_original_tier"] = "very_near_original"
            elif near_original_flag:
                record["near_original_tier"] = "near_original"
            else:
                record["near_original_tier"] = "distinct_from_original"
        else:
            record["near_original_tier"] = (
                "near_original" if near_original_flag else "distinct_from_original"
            )
        if near_original_flag:
            near_original_candidate_indices.append(k)
            if record.get("decision") == "kept":
                near_original_threshold_indices.append(k)
    boundary_diagnostics["near_original_candidate_indices"] = [
        int(k) for k in sorted(set(near_original_candidate_indices))
    ]
    boundary_diagnostics["near_original_candidate_sigmas"] = [
        float(sigmas[k]) for k in sorted(set(near_original_candidate_indices))
    ]
    boundary_diagnostics["near_original_threshold_indices"] = [
        int(k) for k in sorted(set(near_original_threshold_indices))
    ]
    boundary_diagnostics["near_original_threshold_sigmas"] = [
        float(sigmas[k]) for k in sorted(set(near_original_threshold_indices))
    ]

    result = {
        "ls_equal": ls_eq,
        "step_m": float(step_m),
        "width_m": None if width_m is None else float(width_m),
        "distance_sampling": distance_sampling,
        "sigma_min": float(sigmas[0]),
        "sigma_max": float(sigmas[-1]),
        "sigma_min_requested": None if sigma_min is None else float(sigma_min),
        "sigma_max_requested": None if sigma_max is None else float(sigma_max),
        "sigmas": sigmas,
        "metrics": metrics,
        "score": score,
        "threshold_indices": final_threshold_indices,
        "threshold_sigmas": threshold_sigmas,
        "boundary_indices": final_threshold_indices,
        "boundary_sigmas": threshold_sigmas,
        "score_peak_candidate_indices": score_peak_candidate_indices,
        "score_peak_candidate_sigmas": score_peak_candidate_sigmas,
        "rejected_peak_candidate_indices": rejected_peak_candidate_indices,
        "rejected_peak_candidate_sigmas": rejected_peak_candidate_sigmas,
        "near_original_candidate_indices": boundary_diagnostics["near_original_candidate_indices"],
        "near_original_candidate_sigmas": np.asarray(
            boundary_diagnostics["near_original_candidate_sigmas"], float
        ),
        "near_original_threshold_indices": boundary_diagnostics["near_original_threshold_indices"],
        "near_original_threshold_sigmas": np.asarray(
            boundary_diagnostics["near_original_threshold_sigmas"], float
        ),
        "score_peak_threshold_indices": peak_threshold_indices,
        "score_peak_threshold_sigmas": score_peak_threshold_sigmas,
        "peak_boundary_indices": peak_threshold_indices,
        "peak_boundary_sigmas": score_peak_threshold_sigmas,
        "heuristic_threshold_indices": heuristic_threshold_indices,
        "heuristic_threshold_sigmas": heuristic_threshold_sigmas,
        "added_boundary_indices": heuristic_threshold_indices,
        "added_boundary_sigmas": heuristic_threshold_sigmas,
        "terminal_threshold_indices": terminal_threshold_indices,
        "terminal_threshold_sigmas": terminal_threshold_sigmas,
        "mode_candidate_sigmas": stable_mode_candidate_sigmas,
        "stable_mode_candidate_sigmas": stable_mode_candidate_sigmas,
        "stable_mode_sigmas": stable_mode_sigmas,
        "stable_modes": stable_modes,
        "stable_curvature_sign_changes": stable_mode_sign_changes,
        "stable_curvature_lobes": stable_mode_lobes,
        "stable_mode_labels": stable_mode_labels,
        "threshold_mode_sigmas": threshold_mode_sigmas,
        "threshold_modes": threshold_modes,
        "threshold_curvature_sign_changes": threshold_mode_sign_changes,
        "threshold_curvature_lobes": threshold_mode_lobes,
        "threshold_mode_labels": threshold_mode_labels,
        "discarded_mode_candidate_sigmas": discarded_mode_candidate_sigmas,
        "boundary_diagnostics": boundary_diagnostics,
        "axis_fallback_used": axis_info["axis_fallback_used"],
        "axis_info": axis_info,
        "mid_info": mid_info,
        "terminal_info": terminal_info,
        "prune_diagnostics": prune_diagnostics,
        "use_threshold_sigmas_as_modes": bool(use_threshold_sigmas_as_modes),
        "mode_sigma_source": mode_sigma_source,
        "mode_sigmas": mode_sigmas,
        "modes": modes,
        "curvature_sign_changes": sign_changes,
        "curvature_lobes": lobes,
        "mode_labels": labels,
        "plot_data": plot_data,
    }

    if make_plots:
        from supporting_plotting import plot_modes_from_result, plot_thresholding_from_result

        plot_thresholding_from_result(result)

        if derive_modes and len(modes):
            plot_modes_from_result(result)

    return result


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
