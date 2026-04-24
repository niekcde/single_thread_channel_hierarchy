"""Boundary detection and mode-selection helpers for the main pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString

from supporting_geometry import simplify_mode, snap_vertices_to_original
from supporting_metrics import (
    curvature_sign_changes,
    global_sinuosity,
    mean_distance_to_original,
    turning_energy,
)


def moving_average(x, w: int = 7):
    x = np.asarray(x, float)
    if w <= 1:
        return x

    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")


def _norm01(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def local_jump(y, k: int, r: int = 6) -> float:
    """Difference of neighborhood means around index k."""
    y = np.asarray(y, float)
    left = np.mean(y[max(0, k - r) : k]) if k > 0 else y[0]
    right = np.mean(y[k : min(len(y), k + r)])
    return float(abs(right - left))


def find_local_maxima(y):
    y = np.asarray(y, float)
    if len(y) < 3:
        return np.array([], dtype=int)

    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def find_local_minima(y):
    y = np.asarray(y, float)
    if len(y) < 3:
        return np.array([], dtype=int)

    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1


def peak_prominence(score, idx: int, left_right: int = 12) -> float:
    score = np.asarray(score, float)
    lo = max(0, idx - left_right)
    hi = min(len(score), idx + left_right + 1)
    valley = np.min(score[lo:hi])
    return float(score[idx] - valley)


def valley_bounded_peak_stats(score, peak_idx: int, valleys):
    score = np.asarray(score, float)
    valleys = np.asarray(valleys, dtype=int)

    pos = int(np.searchsorted(valleys, peak_idx))
    left_valley = int(valleys[pos - 1]) if pos > 0 else None
    right_valley = int(valleys[pos]) if pos < len(valleys) else None

    if left_valley is None and right_valley is None:
        base = np.nan
        prominence = np.nan
        contrast = np.nan
        prom_frac = np.nan
        log_ratio = np.nan
    else:
        if left_valley is None:
            base = float(score[right_valley])
        elif right_valley is None:
            base = float(score[left_valley])
        else:
            base = float(max(score[left_valley], score[right_valley]))

        peak = float(score[peak_idx])
        prominence = float(peak - base)
        contrast = float(prominence / (peak + base + 1e-12))
        prom_frac = float(prominence / (peak + 1e-12))
        log_ratio = float(np.log((peak + 1e-12) / (base + 1e-12)))

    return {
        "left_valley_score_idx": left_valley,
        "right_valley_score_idx": right_valley,
        "base_score": None if not np.isfinite(base) else float(base),
        "valley_prominence": None if not np.isfinite(prominence) else float(prominence),
        "contrast": None if not np.isfinite(contrast) else float(contrast),
        "prom_frac": None if not np.isfinite(prom_frac) else float(prom_frac),
        "log_ratio": None if not np.isfinite(log_ratio) else float(log_ratio),
    }


def robust_prominence_floor(prominences, mad_mult: float = 1.5) -> float:
    prominences = np.asarray(prominences, float)
    prominences = prominences[np.isfinite(prominences) & (prominences > 0)]
    if len(prominences) == 0:
        return 0.0

    median = float(np.median(prominences))
    mad = float(np.median(np.abs(prominences - median)))
    return float(median + mad_mult * mad)


def build_boundary_score(sigmas, sinuosity, turning, dist, smooth_w: int = 7):
    s1 = moving_average(_norm01(sinuosity), w=smooth_w)
    s2 = moving_average(_norm01(turning), w=smooth_w)
    s3 = moving_average(_norm01(dist), w=smooth_w)
    return np.abs(np.diff(s1, n=2)) + np.abs(np.diff(s2, n=2)) + np.abs(
        np.diff(s3, n=2)
    )


def detect_boundary_thresholds_autoK(
    sigmas,
    sinuosity,
    turning,
    dist,
    *,
    selection_mode: str = "adaptive",
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
    dist_jump_min: float = 50.0,
    turn_jump_min: float = 0.00015,
    sinu_jump_rel: float = 0.02,
    dist_jump_rel: float = 0.03,
    turn_jump_rel: float = 0.05,
    max_boundaries: int = 6,
    gap_ratio_stop: float = 1.8,
    return_diagnostics: bool = False,
):
    sigmas = np.asarray(sigmas, float)
    score = build_boundary_score(sigmas, sinuosity, turning, dist, smooth_w=smooth_w)
    diagnostics = {
        "selection_mode": str(selection_mode),
        "use_score_percentile_gate": bool(use_score_percentile_gate),
        "score_percentile": float(score_percentile),
        "prominence_percentile": float(prominence_percentile),
        "score_threshold": None,
        "prominence_threshold": None,
        "prominence_floor": None,
        "prominence_mad_mult": float(prominence_mad_mult),
        "contrast_low": float(contrast_low),
        "contrast_high": float(contrast_high),
        "borderline_prominence_mult": float(borderline_prominence_mult),
        "borderline_prominence_floor": None,
        "effective_jump_thresholds": {},
        "min_sep": int(min_sep),
        "gap_ratio_stop": float(gap_ratio_stop),
        "candidates": [],
    }

    def finish(boundary_idx):
        boundary_idx = list(boundary_idx)
        diagnostics["kept_indices"] = [int(k) for k in boundary_idx]
        diagnostics["kept_sigmas"] = [float(sigmas[k]) for k in boundary_idx]
        diagnostics["candidates"] = sorted(
            diagnostics["candidates"], key=lambda item: item["sigma_idx"]
        )
        result = (boundary_idx, sigmas[boundary_idx], score)
        if return_diagnostics:
            return result + (diagnostics,)
        return result

    peaks = find_local_maxima(score)
    valleys = find_local_minima(score)
    peak_records = {}
    for peak in peaks:
        k = int(peak + 1)
        peak_stats = valley_bounded_peak_stats(score, int(peak), valleys)
        record = {
            "peak_score_idx": int(peak),
            "sigma_idx": k,
            "sigma": float(sigmas[k]),
            "score": float(score[peak]),
            "prominence": None,
            "valley_prominence": peak_stats["valley_prominence"],
            "base_score": peak_stats["base_score"],
            "contrast": peak_stats["contrast"],
            "prom_frac": peak_stats["prom_frac"],
            "log_ratio": peak_stats["log_ratio"],
            "left_valley_score_idx": peak_stats["left_valley_score_idx"],
            "right_valley_score_idx": peak_stats["right_valley_score_idx"],
            "local_jump_sinuosity": None,
            "local_jump_distance": None,
            "local_jump_turning": None,
            "passes_score_threshold": not use_score_percentile_gate,
            "passes_prominence_threshold": False,
            "passes_jump_gate": None,
            "passes_contrast_gate": None,
            "contrast_band": None,
            "decision": "unprocessed",
            "reason": None,
        }
        peak_records[int(peak)] = record
        diagnostics["candidates"].append(record)

    if len(peaks) == 0:
        return finish([])

    if use_score_percentile_gate:
        score_cut = np.percentile(score, score_percentile)
        diagnostics["score_threshold"] = float(score_cut)
        peaks = peaks[score[peaks] >= score_cut]
        score_kept = set(int(peak) for peak in peaks)
        for peak, record in peak_records.items():
            record["passes_score_threshold"] = peak in score_kept
            if peak not in score_kept:
                record["decision"] = "rejected"
                record["reason"] = "below score percentile"
        if len(peaks) == 0:
            return finish([])
    else:
        diagnostics["score_threshold"] = None

    if selection_mode == "adaptive":
        prom = np.array(
            [
                peak_records[int(peak)]["valley_prominence"]
                if peak_records[int(peak)]["valley_prominence"] is not None
                else np.nan
                for peak in peaks
            ],
            float,
        )
        prom_cut = robust_prominence_floor(prom, mad_mult=prominence_mad_mult)
        diagnostics["prominence_floor"] = float(prom_cut)
        diagnostics["prominence_threshold"] = float(prom_cut)
        borderline_floor = float(borderline_prominence_mult * prom_cut)
        diagnostics["borderline_prominence_floor"] = borderline_floor

        keep_mask = np.zeros(len(peaks), dtype=bool)
        for i, (peak, peak_prom) in enumerate(zip(peaks, prom)):
            record = peak_records[int(peak)]
            record["prominence"] = None if not np.isfinite(peak_prom) else float(peak_prom)
            record["passes_prominence_threshold"] = bool(
                np.isfinite(peak_prom) and peak_prom >= prom_cut
            )
            if not np.isfinite(peak_prom) or peak_prom < prom_cut:
                record["decision"] = "rejected"
                record["reason"] = "below adaptive prominence floor"
                continue
            keep_mask[i] = True

        peaks, prom = peaks[keep_mask], prom[keep_mask]
        if len(peaks) == 0:
            return finish([])
    else:
        prom = np.array([peak_prominence(score, p, left_right=left_right) for p in peaks])
        prom_cut = np.percentile(prom, prominence_percentile)
        diagnostics["prominence_threshold"] = float(prom_cut)
        for peak, peak_prom in zip(peaks, prom):
            record = peak_records[int(peak)]
            record["prominence"] = float(peak_prom)
            record["passes_prominence_threshold"] = bool(peak_prom >= prom_cut)
            if peak_prom < prom_cut:
                record["decision"] = "rejected"
                record["reason"] = "below prominence percentile"
        keep = prom >= prom_cut
        peaks, prom = peaks[keep], prom[keep]
        if len(peaks) == 0:
            return finish([])

    order = np.argsort(prom)[::-1]
    peaks, prom = peaks[order], prom[order]

    if use_relative_jumps:
        sinu_scale = max(np.ptp(sinuosity), 1e-12)
        dist_scale = max(np.max(dist), 1e-12)
        turn_scale = max(np.ptp(turning), 1e-12)
        sinu_jump_min_eff = max(sinu_jump_min, sinu_jump_rel * sinu_scale)
        dist_jump_min_eff = max(dist_jump_min, dist_jump_rel * dist_scale)
        turn_jump_min_eff = max(turn_jump_min, turn_jump_rel * turn_scale)
    else:
        sinu_jump_min_eff = sinu_jump_min
        dist_jump_min_eff = dist_jump_min
        turn_jump_min_eff = turn_jump_min
    diagnostics["effective_jump_thresholds"] = {
        "sinuosity": float(sinu_jump_min_eff),
        "distance": float(dist_jump_min_eff),
        "turning": float(turn_jump_min_eff),
    }

    chosen_k = []
    chosen_prom = []
    for peak, peak_prom in zip(peaks, prom):
        record = peak_records[int(peak)]
        k = int(peak + 1)
        sinu_jump = local_jump(sinuosity, k)
        dist_jump = local_jump(dist, k)
        turn_jump = local_jump(turning, k)
        record["local_jump_sinuosity"] = float(sinu_jump)
        record["local_jump_distance"] = float(dist_jump)
        record["local_jump_turning"] = float(turn_jump)

        passes_jump_gate = not (
            sinu_jump < sinu_jump_min_eff
            and dist_jump < dist_jump_min_eff
            and turn_jump < turn_jump_min_eff
        )
        record["passes_jump_gate"] = bool(passes_jump_gate)

        if selection_mode == "adaptive":
            contrast = record["contrast"]
            if contrast is None or not np.isfinite(contrast):
                record["passes_contrast_gate"] = False
                record["contrast_band"] = "undefined"
                record["decision"] = "rejected"
                record["reason"] = "undefined contrast"
                continue

            if contrast >= contrast_high:
                record["contrast_band"] = "high"
                passes_contrast_gate = passes_jump_gate
                contrast_reason = "insufficient local jump"
            elif contrast >= contrast_low:
                record["contrast_band"] = "mid"
                passes_contrast_gate = passes_jump_gate or peak_prom >= borderline_floor
                contrast_reason = "borderline contrast without strong support"
            else:
                record["contrast_band"] = "low"
                passes_contrast_gate = passes_jump_gate and peak_prom >= borderline_floor
                contrast_reason = "low contrast"

            record["passes_contrast_gate"] = bool(passes_contrast_gate)
            if not passes_contrast_gate:
                record["decision"] = "rejected"
                record["reason"] = contrast_reason
                continue
        elif not passes_jump_gate:
            record["decision"] = "rejected"
            record["reason"] = "insufficient local jump"
            continue

        too_close_to = next(
            (int(chosen) for chosen in chosen_k if abs(k - chosen) < min_sep),
            None,
        )
        if too_close_to is not None:
            record["decision"] = "rejected"
            record["reason"] = f"too close to stronger peak at sigma_idx={too_close_to}"
            continue

        if all(abs(k - chosen) >= min_sep for chosen in chosen_k):
            chosen_k.append(k)
            chosen_prom.append(float(peak_prom))
            record["decision"] = "chosen_before_gap_stop"
            record["reason"] = "passed local filters"

        if len(chosen_k) >= max_boundaries:
            break

    if len(chosen_k) == 0:
        return finish([])

    sort_desc = np.argsort(chosen_prom)[::-1]
    chosen_k = [chosen_k[i] for i in sort_desc]
    chosen_prom = [chosen_prom[i] for i in sort_desc]

    chosen_lookup = {int(record["sigma_idx"]): record for record in diagnostics["candidates"]}
    kept = [chosen_k[0]]
    chosen_lookup[int(chosen_k[0])]["decision"] = "kept"
    chosen_lookup[int(chosen_k[0])]["reason"] = "selected boundary"
    for i in range(1, len(chosen_k)):
        if chosen_prom[i - 1] / max(chosen_prom[i], 1e-12) >= gap_ratio_stop:
            chosen_lookup[int(chosen_k[i])]["decision"] = "rejected"
            chosen_lookup[int(chosen_k[i])]["reason"] = "weaker than gap-ratio stop"
            break
        kept.append(chosen_k[i])
        chosen_lookup[int(chosen_k[i])]["decision"] = "kept"
        chosen_lookup[int(chosen_k[i])]["reason"] = "selected boundary"

    kept = sorted(kept)
    for chosen in chosen_k:
        if chosen in kept:
            continue
        record = chosen_lookup[int(chosen)]
        if record["decision"] == "chosen_before_gap_stop":
            record["decision"] = "rejected"
            record["reason"] = "weaker than gap-ratio stop"
    return finish(kept)


def detect_mode_boundaries_autoK(*args, **kwargs):
    """Backward-compatible alias for the threshold detector."""
    return detect_boundary_thresholds_autoK(*args, **kwargs)


def maybe_add_terminal_threshold(
    sigmas,
    score,
    boundary_indices,
    *,
    sinuosity,
    turning,
    dist,
    tail_frac: float = 0.20,
    min_prom_frac: float = 0.35,
    min_sep: int = 10,
    log_sigma_tol: float = 0.08,
    left_right: int = 12,
    score_percentile: float = 80,
    use_relative_jumps: bool = True,
    sinu_jump_min: float = 0.05,
    dist_jump_min: float = 50.0,
    turn_jump_min: float = 0.00015,
    sinu_jump_rel: float = 0.02,
    dist_jump_rel: float = 0.03,
    turn_jump_rel: float = 0.05,
    min_sinu_drop: float = 0.03,
    min_dist_increase: float = 0.10,
):
    info = {
        "added_terminal_threshold": False,
        "reason": None,
        "tail_frac": float(tail_frac),
        "min_prom_frac": float(min_prom_frac),
        "min_sep": int(min_sep),
        "log_sigma_tol": float(log_sigma_tol),
        "score_percentile": float(score_percentile),
        "min_sinu_drop": float(min_sinu_drop),
        "min_dist_increase": float(min_dist_increase),
    }

    sigmas = np.asarray(sigmas, float)
    score = np.asarray(score, float)
    sinuosity = np.asarray(sinuosity, float)
    turning = np.asarray(turning, float)
    dist = np.asarray(dist, float)
    boundary_indices = sorted(set(int(k) for k in boundary_indices))
    info["boundary_indices_before"] = [int(k) for k in boundary_indices]

    if len(sigmas) < 10 or len(score) < 5:
        info["reason"] = "insufficient data"
        return boundary_indices, info

    score_cut = float(np.percentile(score, score_percentile))
    info["score_threshold"] = score_cut

    j0 = max(int((1 - tail_frac) * len(score)), 0)
    tail = score[j0:]
    info["tail_start_score_idx"] = int(j0)
    info["tail_start_sigma"] = float(sigmas[min(j0 + 1, len(sigmas) - 1)])
    if len(tail) < 5:
        info["reason"] = "tail too short"
        return boundary_indices, info

    j_peak = j0 + int(np.argmax(tail))
    k = j_peak + 1
    info["tail_peak_score_idx"] = int(j_peak)
    info["sigma_idx"] = int(k)
    info["sigma"] = float(sigmas[k])
    info["score"] = float(score[j_peak])
    info["passes_score_threshold"] = bool(score[j_peak] >= score_cut)

    if not info["passes_score_threshold"]:
        info["reason"] = "tail score below percentile threshold"
        return boundary_indices, info

    prom = peak_prominence(score, j_peak, left_right=left_right)
    prom_global = float(
        np.max([peak_prominence(score, j, left_right=left_right) for j in range(len(score))])
    )
    info["prominence"] = float(prom)
    info["prominence_global_max"] = float(prom_global)
    info["prominence_ratio"] = float(prom / max(prom_global, 1e-12))
    if prom_global <= 0:
        info["reason"] = "no global prominence"
        return boundary_indices, info

    if prom < min_prom_frac * prom_global:
        info["reason"] = f"tail peak not prominent enough ({prom:.4g})"
        return boundary_indices, info

    if any(abs(k - existing) < min_sep for existing in boundary_indices):
        info["reason"] = "too close to existing threshold (index)"
        return boundary_indices, info

    if any(abs(np.log(sigmas[k] / sigmas[existing])) < log_sigma_tol for existing in boundary_indices):
        info["reason"] = "too close to existing threshold (log-sigma)"
        return boundary_indices, info

    if use_relative_jumps:
        sinu_scale = max(np.ptp(sinuosity), 1e-12)
        dist_scale = max(np.max(dist), 1e-12)
        turn_scale = max(np.ptp(turning), 1e-12)
        sinu_jump_min_eff = max(sinu_jump_min, sinu_jump_rel * sinu_scale)
        dist_jump_min_eff = max(dist_jump_min, dist_jump_rel * dist_scale)
        turn_jump_min_eff = max(turn_jump_min, turn_jump_rel * turn_scale)
    else:
        sinu_jump_min_eff = sinu_jump_min
        dist_jump_min_eff = dist_jump_min
        turn_jump_min_eff = turn_jump_min
    info["effective_jump_thresholds"] = {
        "sinuosity": float(sinu_jump_min_eff),
        "distance": float(dist_jump_min_eff),
        "turning": float(turn_jump_min_eff),
    }

    sinu_jump = local_jump(sinuosity, k)
    dist_jump = local_jump(dist, k)
    turn_jump = local_jump(turning, k)
    passes_jump_gate = not (
        sinu_jump < sinu_jump_min_eff
        and dist_jump < dist_jump_min_eff
        and turn_jump < turn_jump_min_eff
    )
    info["local_jump_sinuosity"] = float(sinu_jump)
    info["local_jump_distance"] = float(dist_jump)
    info["local_jump_turning"] = float(turn_jump)
    info["passes_jump_gate"] = bool(passes_jump_gate)

    ref_k = boundary_indices[-1] if len(boundary_indices) else min(j0 + 1, len(sigmas) - 1)
    sinu_ref = float(sinuosity[ref_k])
    sinu_candidate = float(sinuosity[k])
    dist_ref = float(dist[ref_k])
    dist_candidate = float(dist[k])
    turn_ref = float(turning[ref_k])
    turn_candidate = float(turning[k])

    sinu_drop = float(sinu_ref - sinu_candidate)
    dist_ratio = float(dist_candidate / max(dist_ref, 1e-9))
    turn_drop = float((turn_ref - turn_candidate) / max(turn_ref, 1e-12))
    metric_delta_pass = (
        sinu_drop >= float(min_sinu_drop)
        and dist_ratio >= float(1.0 + min_dist_increase)
    )
    info["reference_sigma_idx"] = int(ref_k)
    info["reference_sigma"] = float(sigmas[ref_k])
    info["sinuosity_reference"] = sinu_ref
    info["sinuosity_candidate"] = sinu_candidate
    info["sinuosity_drop"] = sinu_drop
    info["distance_reference"] = dist_ref
    info["distance_candidate"] = dist_candidate
    info["distance_ratio"] = dist_ratio
    info["turning_reference"] = turn_ref
    info["turning_candidate"] = turn_candidate
    info["turning_drop_frac"] = turn_drop
    info["passes_metric_delta_gate"] = bool(metric_delta_pass)

    if not passes_jump_gate and not metric_delta_pass:
        info["reason"] = "insufficient local jump and tail metric change"
        return boundary_indices, info

    out_idx = sorted(set(boundary_indices + [int(k)]))
    info["added_terminal_threshold"] = True
    info["reason"] = "accepted"
    info["boundary_indices_after"] = [int(v) for v in out_idx]
    return out_idx, info


def representative_modes_by_stability(sigmas, reps, boundaries_idx, score):
    sigmas = np.asarray(sigmas, float)
    bounds = [0] + list(boundaries_idx) + [len(sigmas) - 1]
    out_sig = []
    out_lines = []

    # score index j corresponds to sigma index j + 1.
    for a, b in zip(bounds[:-1], bounds[1:]):
        lo = max(a + 2, 2)
        hi = min(b - 2, len(sigmas) - 3)
        if hi <= lo:
            idx = int((a + b) // 2)
        else:
            j_lo, j_hi = lo - 1, hi - 1
            j = j_lo + int(np.argmin(score[j_lo : j_hi + 1]))
            idx = j + 1

        out_sig.append(sigmas[idx])
        out_lines.append(reps[idx])

    return np.array(out_sig, float), out_lines


def pick_axis_sigma(
    sigmas,
    sinu,
    turn,
    dist,
    sinu_target: float = 1.05,
    turn_frac: float = 0.15,
    dist_frac: float = 0.80,
):
    sigmas = np.asarray(sigmas, float)
    sinu = np.asarray(sinu, float)
    turn = np.asarray(turn, float)
    dist = np.asarray(dist, float)

    idx = np.where(sinu <= sinu_target)[0]
    if len(idx) > 0:
        return float(sigmas[idx[0]])

    turn0 = float(turn[0])
    idx = np.where(turn <= turn_frac * turn0)[0]
    if len(idx) > 0:
        return float(sigmas[idx[0]])

    dist_max = float(np.max(dist))
    idx = np.where(dist >= dist_frac * dist_max)[0]
    if len(idx) > 0:
        return float(sigmas[idx[0]])

    return float(sigmas[-1])


def prune_redundant_adjacent_modes(
    modes,
    mode_sigmas,
    *,
    original=None,
    width_m=None,
    dist_abs: float = 30.0,
    first_dist_w_mult: float = 2.0,
    adj_dist_w_mult: float = 1.0,
    dist_w_frac: float = 0.30,
    turn_frac_min: float = 0.15,
    sc_frac_min: float = 0.25,
    sinu_abs_min: float = 0.0,
    dist_sample_spacing: float | None = None,
    dist_min_samples: int = 200,
    dist_max_samples: int = 4000,
    verbose: bool = False,
    return_diagnostics: bool = False,
):
    """Drop adjacent modes that are too similar to the previous kept mode."""
    if len(modes) <= 1:
        diagnostics = {
            "settings": {
                "dist_abs": float(dist_abs),
                "dist_w_frac": float(dist_w_frac),
                "turn_frac_min": float(turn_frac_min),
                "sc_frac_min": float(sc_frac_min),
                "sinu_abs_min": float(sinu_abs_min),
                "dist_sample_spacing": None
                if dist_sample_spacing is None
                else float(dist_sample_spacing),
                "dist_min_samples": int(dist_min_samples),
                "dist_max_samples": int(dist_max_samples),
            },
            "candidates": [],
            "fallback": {"used": False},
        }
        if return_diagnostics:
            return modes, mode_sigmas, diagnostics
        return modes, mode_sigmas

    def dist_thresh(mult):
        if width_m is not None:
            return float(mult) * float(width_m)
        return float(dist_abs)

    keep_modes = []
    keep_sig = []
    diagnostics = {
        "settings": {
            "dist_abs": float(dist_abs),
            "dist_w_frac": float(dist_w_frac),
            "turn_frac_min": float(turn_frac_min),
            "sc_frac_min": float(sc_frac_min),
            "sinu_abs_min": float(sinu_abs_min),
            "dist_sample_spacing": None
            if dist_sample_spacing is None
            else float(dist_sample_spacing),
            "dist_min_samples": int(dist_min_samples),
            "dist_max_samples": int(dist_max_samples),
        },
        "candidates": [],
        "fallback": {"used": False},
    }
    for i, (mode, sigma) in enumerate(zip(modes, mode_sigmas)):
        record = {
            "candidate_index": int(i),
            "sigma": float(sigma),
            "comparison_basis": None,
            "distance": None,
            "distance_threshold": None,
            "distance_pass": None,
            "turn_drop_frac": None,
            "sign_change_drop_frac": None,
            "sinuosity_drop_abs": None,
            "kept": False,
            "reason": None,
        }
        if len(keep_modes) == 0:
            if original is None:
                keep_modes.append(mode)
                keep_sig.append(float(sigma))
                record["comparison_basis"] = "none"
                record["kept"] = True
                record["reason"] = "kept without comparison baseline"
                diagnostics["candidates"].append(record)
                continue

            prev = original
            distance, sampling = mean_distance_to_original(
                prev,
                mode,
                sample_spacing=dist_sample_spacing,
                min_samples=dist_min_samples,
                max_samples=dist_max_samples,
                return_diagnostics=True,
            )
            distance_thresh = dist_thresh(first_dist_w_mult)
            record["comparison_basis"] = "original"
        else:
            prev = keep_modes[-1]
            distance, sampling = mean_distance_to_original(
                prev,
                mode,
                sample_spacing=dist_sample_spacing,
                min_samples=dist_min_samples,
                max_samples=dist_max_samples,
                return_diagnostics=True,
            )
            distance_thresh = dist_thresh(adj_dist_w_mult)
            record["comparison_basis"] = "previous_kept_mode"

        record["distance"] = float(distance)
        record["distance_threshold"] = float(distance_thresh)
        record["distance_pass"] = bool(distance >= distance_thresh)
        record["distance_sampling"] = sampling

        if verbose:
            print(f"i={i} distance={distance:.2f} threshold={distance_thresh:.2f}")

        if distance < distance_thresh:
            record["reason"] = "failed distance gate"
            diagnostics["candidates"].append(record)
            continue

        prev_turning = turning_energy(prev)
        mode_turning = turning_energy(mode)
        turn_drop = (prev_turning - mode_turning) / max(prev_turning, 1e-12)

        prev_sign_changes = curvature_sign_changes(prev)
        mode_sign_changes = curvature_sign_changes(mode)
        sc_drop = (prev_sign_changes - mode_sign_changes) / max(prev_sign_changes, 1)
        prev_sinuosity = global_sinuosity(prev)
        mode_sinuosity = global_sinuosity(mode)
        sinu_drop = prev_sinuosity - mode_sinuosity
        record["turn_drop_frac"] = float(turn_drop)
        record["sign_change_drop_frac"] = float(sc_drop)
        record["sinuosity_drop_abs"] = float(sinu_drop)

        if verbose:
            print(f"i={i} turn_drop={turn_drop:.3f} sc_drop={sc_drop:.3f}")

        if (
            (turn_drop >= turn_frac_min)
            or (sc_drop >= sc_frac_min)
            or (sinu_drop >= sinu_abs_min)
        ):
            if verbose:
                print(f"keeping mode i={i}")
            keep_modes.append(mode)
            keep_sig.append(float(sigma))
            record["kept"] = True
            record["reason"] = "passed complexity gate"
        else:
            record["reason"] = "failed complexity gate"
        diagnostics["candidates"].append(record)

    if len(keep_modes) == 0:
        keep_modes = [modes[-1]]
        keep_sig = [float(mode_sigmas[-1])]
        diagnostics["fallback"] = {
            "used": True,
            "sigma": float(mode_sigmas[-1]),
            "reason": "no candidate survived pruning",
        }

    if return_diagnostics:
        return keep_modes, np.asarray(keep_sig, float), diagnostics
    return keep_modes, np.asarray(keep_sig, float)


def pick_mid_index_by_max_slope(sigmas, y) -> int:
    sigmas = np.asarray(sigmas, float)
    y = np.asarray(y, float)
    x = np.log(sigmas)
    dy = np.gradient(y, x)
    return int(np.argmax(np.abs(dy)))


def try_insert_mid_boundary(
    idx: List[int],
    *,
    sigmas: np.ndarray,
    reps: List[LineString],
    score: np.ndarray,
    ls_eq: LineString,
    width_m=None,
    min_sep: int = 10,
    log_sigma_tol: float = 0.08,
    choose_interval: str = "largest",
    eps_floor: float = 10.0,
    eps_factor: float = 1.0,
    eps_power: float = 0.5,
    snap_to_original: bool = False,
    dist_abs: float = 10.0,
    dist_w_frac: float = 0.30,
    first_dist_w_mult: float = 1.0,
    adj_dist_w_mult: float = 0.7,
    turn_frac_min: float = 0.15,
    sc_frac_min: float = 0.25,
    sinu_abs_min: float = 0.0,
    dist_sample_spacing: float | None = None,
    dist_min_samples: int = 200,
    dist_max_samples: int = 4000,
) -> Tuple[List[int], Dict[str, Any]]:
    """Attempt to add one mid boundary if it survives pruning."""
    idx = sorted(set(int(k) for k in idx))
    n = len(sigmas)

    if n < 10:
        return idx, {"mid_added": False, "reason": "too few sigmas"}

    bounds = [0] + idx + [n - 1]
    intervals = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= 2 * min_sep:
            continue
        span = abs(np.log(sigmas[b]) - np.log(sigmas[a] + 1e-12))
        intervals.append((span, a, b))

    if not intervals:
        return idx, {"mid_added": False, "reason": "no interval large enough for mid"}

    if choose_interval == "worst_score":
        j_peak = int(np.argmax(score))
        k_peak = j_peak + 1
        chosen = None
        for _, a, b in intervals:
            if a < k_peak < b:
                chosen = (a, b)
                break

        if chosen is None:
            _, a, b = max(intervals, key=lambda t: t[0])
        else:
            a, b = chosen
    else:
        _, a, b = max(intervals, key=lambda t: t[0])

    lo = max(a + min_sep, 2)
    hi = min(b - min_sep, n - 3)
    if hi <= lo:
        return idx, {"mid_added": False, "reason": "interval too small after margins"}

    j_lo = lo - 1
    j_hi = hi - 1
    j_mid = j_lo + int(np.argmin(score[j_lo : j_hi + 1]))
    k_mid = j_mid + 1

    if any(abs(k_mid - k) < min_sep for k in idx):
        return idx, {
            "mid_added": False,
            "reason": "mid too close (index)",
            "candidate_sigma": float(sigmas[int(k_mid)]),
        }

    sigma_mid = float(sigmas[k_mid])
    for k in idx:
        sigma_k = float(sigmas[k])
        if abs(np.log(sigma_mid / sigma_k)) < log_sigma_tol:
            return idx, {
                "mid_added": False,
                "reason": "mid too close (log-sigma)",
                "candidate_sigma": float(sigmas[int(k_mid)]),
            }

    def build_modes_from_idx(idx_list: List[int]) -> Tuple[np.ndarray, List[LineString]]:
        mode_sigmas, mode_lines = representative_modes_by_stability(
            sigmas, reps, idx_list, score
        )

        modes = []
        for sigma, mode in zip(mode_sigmas, mode_lines):
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
            dist_abs=float(dist_abs),
            dist_w_frac=float(dist_w_frac),
            first_dist_w_mult=float(first_dist_w_mult),
            adj_dist_w_mult=float(adj_dist_w_mult),
            turn_frac_min=float(turn_frac_min),
            sc_frac_min=float(sc_frac_min),
            sinu_abs_min=float(sinu_abs_min),
            dist_sample_spacing=dist_sample_spacing,
            dist_min_samples=int(dist_min_samples),
            dist_max_samples=int(dist_max_samples),
        )
        return mode_sigmas, modes

    _, base_modes = build_modes_from_idx(idx)
    idx_try = sorted(set(idx + [k_mid]))
    _, try_modes = build_modes_from_idx(idx_try)

    if len(try_modes) > len(base_modes):
        return idx_try, {
            "mid_added": True,
            "k_mid": int(k_mid),
            "sigma_mid": float(sigmas[int(k_mid)]),
            "interval": (int(a), int(b)),
            "base_mode_count": int(len(base_modes)),
            "try_mode_count": int(len(try_modes)),
        }

    return idx, {
        "mid_added": False,
        "reason": "redundant after pruning",
        "candidate_sigma": float(sigmas[int(k_mid)]),
        "interval": (int(a), int(b)),
        "base_mode_count": int(len(base_modes)),
        "try_mode_count": int(len(try_modes)),
    }


def maybe_add_terminal_mode(
    sigmas,
    reps,
    modes,
    mode_sigmas,
    score,
    *,
    tail_frac: float = 0.20,
    min_prom_frac: float = 0.35,
    min_sinu_drop: float = 0.03,
    min_dist_increase: float = 0.10,
    ls_eq=None,
    dist_sample_spacing: float | None = None,
    dist_min_samples: int = 200,
    dist_max_samples: int = 4000,
):
    info = {
        "added_terminal": False,
        "reason": None,
        "tail_frac": float(tail_frac),
        "min_prom_frac": float(min_prom_frac),
        "min_sinu_drop": float(min_sinu_drop),
        "min_dist_increase": float(min_dist_increase),
    }
    if len(sigmas) < 10 or len(score) < 10 or ls_eq is None or len(modes) == 0:
        info["reason"] = "insufficient data"
        return mode_sigmas, modes, info

    sigmas = np.asarray(sigmas, float)
    score = np.asarray(score, float)

    j0 = max(int((1 - tail_frac) * len(score)), 0)
    tail = score[j0:]
    info["tail_start_score_idx"] = int(j0)
    info["tail_start_sigma"] = float(sigmas[min(j0 + 1, len(sigmas) - 1)])
    if len(tail) < 5:
        info["reason"] = "tail too short"
        return mode_sigmas, modes, info

    j_peak = j0 + int(np.argmax(tail))
    info["tail_peak_score_idx"] = int(j_peak)

    prom = peak_prominence(score, j_peak, left_right=12)
    prom_global = float(
        np.max([peak_prominence(score, j, left_right=12) for j in range(1, len(score) - 1)])
    )
    info["prominence"] = float(prom)
    info["prominence_global_max"] = float(prom_global)
    info["prominence_ratio"] = float(prom / max(prom_global, 1e-12))
    if prom_global <= 0:
        info["reason"] = "no global prominence"
        return mode_sigmas, modes, info

    if prom < min_prom_frac * prom_global:
        info["reason"] = f"tail peak not prominent enough ({prom:.4g})"
        return mode_sigmas, modes, info

    k = j_peak + 1
    sigma_terminal = float(sigmas[k])
    candidate = reps[k]
    info["sigma"] = float(sigma_terminal)

    last_mode = modes[-1]
    sinu_last = global_sinuosity(last_mode)
    sinu_candidate = global_sinuosity(candidate)
    info["sinuosity_last"] = float(sinu_last)
    info["sinuosity_candidate"] = float(sinu_candidate)
    info["sinuosity_drop"] = float(sinu_last - sinu_candidate)

    dist_last, sampling = mean_distance_to_original(
        ls_eq,
        last_mode,
        sample_spacing=dist_sample_spacing,
        min_samples=dist_min_samples,
        max_samples=dist_max_samples,
        return_diagnostics=True,
    )
    dist_candidate = mean_distance_to_original(
        ls_eq,
        candidate,
        sample_spacing=dist_sample_spacing,
        min_samples=dist_min_samples,
        max_samples=dist_max_samples,
    )
    info["distance_last"] = float(dist_last)
    info["distance_candidate"] = float(dist_candidate)
    info["distance_ratio"] = float(dist_candidate / max(dist_last, 1e-9))
    info["distance_sampling"] = sampling

    if (sinu_last - sinu_candidate) < min_sinu_drop:
        info["reason"] = "no extra sinuosity drop"
        return mode_sigmas, modes, info

    if dist_candidate < (1 + min_dist_increase) * max(dist_last, 1e-9):
        info["reason"] = "no extra deviation increase"
        return mode_sigmas, modes, info

    mode_sigmas2 = np.append(mode_sigmas, sigma_terminal)
    modes2 = list(modes) + [candidate]
    info["added_terminal"] = True
    info["reason"] = "accepted"
    return mode_sigmas2, modes2, info


def _demo() -> None:
    sigmas = np.geomspace(10, 1000, 40)
    y = np.r_[np.linspace(2, 1.8, 20), np.linspace(1.4, 1.1, 20)]
    turn = np.r_[np.linspace(0.01, 0.008, 20), np.linspace(0.004, 0.001, 20)]
    dist = np.linspace(0, 500, 40)
    idx, boundary_sigmas, score = detect_mode_boundaries_autoK(sigmas, y, turn, dist)

    print("boundary indices:", idx)
    print("boundary sigmas:", np.round(boundary_sigmas, 2).tolist())
    print("score length:", len(score))


if __name__ == "__main__":
    _demo()
