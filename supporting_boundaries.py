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


def peak_prominence(score, idx: int, left_right: int = 12) -> float:
    score = np.asarray(score, float)
    lo = max(0, idx - left_right)
    hi = min(len(score), idx + left_right + 1)
    valley = np.min(score[lo:hi])
    return float(score[idx] - valley)


def build_boundary_score(sigmas, sinuosity, turning, dist, smooth_w: int = 7):
    s1 = moving_average(_norm01(sinuosity), w=smooth_w)
    s2 = moving_average(_norm01(turning), w=smooth_w)
    s3 = moving_average(_norm01(dist), w=smooth_w)
    return np.abs(np.diff(s1, n=2)) + np.abs(np.diff(s2, n=2)) + np.abs(
        np.diff(s3, n=2)
    )


def detect_mode_boundaries_autoK(
    sigmas,
    sinuosity,
    turning,
    dist,
    *,
    min_sep: int = 10,
    smooth_w: int = 7,
    score_percentile: float = 80,
    prominence_percentile: float = 70,
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
):
    sigmas = np.asarray(sigmas, float)
    score = build_boundary_score(sigmas, sinuosity, turning, dist, smooth_w=smooth_w)

    peaks = find_local_maxima(score)
    if len(peaks) == 0:
        return [], np.array([]), score

    score_cut = np.percentile(score, score_percentile)
    peaks = peaks[score[peaks] >= score_cut]
    if len(peaks) == 0:
        return [], np.array([]), score

    prom = np.array([peak_prominence(score, p, left_right=left_right) for p in peaks])
    prom_cut = np.percentile(prom, prominence_percentile)
    keep = prom >= prom_cut
    peaks, prom = peaks[keep], prom[keep]
    if len(peaks) == 0:
        return [], np.array([]), score

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

    chosen_k = []
    chosen_prom = []
    for peak, peak_prom in zip(peaks, prom):
        k = int(peak + 1)

        if (
            local_jump(sinuosity, k) < sinu_jump_min_eff
            and local_jump(dist, k) < dist_jump_min_eff
            and local_jump(turning, k) < turn_jump_min_eff
        ):
            continue

        if all(abs(k - chosen) >= min_sep for chosen in chosen_k):
            chosen_k.append(k)
            chosen_prom.append(float(peak_prom))

        if len(chosen_k) >= max_boundaries:
            break

    if len(chosen_k) == 0:
        return [], np.array([]), score

    sort_desc = np.argsort(chosen_prom)[::-1]
    chosen_k = [chosen_k[i] for i in sort_desc]
    chosen_prom = [chosen_prom[i] for i in sort_desc]

    kept = [chosen_k[0]]
    for i in range(1, len(chosen_k)):
        if chosen_prom[i - 1] / max(chosen_prom[i], 1e-12) >= gap_ratio_stop:
            break
        kept.append(chosen_k[i])

    kept = sorted(kept)
    return kept, sigmas[kept], score


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
    verbose: bool = False,
):
    """Drop adjacent modes that are too similar to the previous kept mode."""
    if len(modes) <= 1:
        return modes, mode_sigmas

    def dist_thresh(mult):
        if width_m is not None:
            return float(mult) * float(width_m)
        return float(dist_abs)

    keep_modes = []
    keep_sig = []
    for i, (mode, sigma) in enumerate(zip(modes, mode_sigmas)):
        if len(keep_modes) == 0:
            if original is None:
                keep_modes.append(mode)
                keep_sig.append(float(sigma))
                continue

            prev = original
            distance = mean_distance_to_original(prev, mode, n=600)
            distance_thresh = dist_thresh(first_dist_w_mult)
        else:
            prev = keep_modes[-1]
            distance = mean_distance_to_original(prev, mode, n=400)
            distance_thresh = dist_thresh(adj_dist_w_mult)

        if verbose:
            print(f"i={i} distance={distance:.2f} threshold={distance_thresh:.2f}")

        if distance < distance_thresh:
            continue

        prev_turning = turning_energy(prev)
        mode_turning = turning_energy(mode)
        turn_drop = (prev_turning - mode_turning) / max(prev_turning, 1e-12)

        prev_sign_changes = curvature_sign_changes(prev)
        mode_sign_changes = curvature_sign_changes(mode)
        sc_drop = (prev_sign_changes - mode_sign_changes) / max(prev_sign_changes, 1)

        if verbose:
            print(f"i={i} turn_drop={turn_drop:.3f} sc_drop={sc_drop:.3f}")

        if (turn_drop >= turn_frac_min) or (sc_drop >= sc_frac_min):
            if verbose:
                print(f"keeping mode i={i}")
            keep_modes.append(mode)
            keep_sig.append(float(sigma))

    if len(keep_modes) == 0:
        keep_modes = [modes[-1]]
        keep_sig = [float(mode_sigmas[-1])]

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
        return idx, {"mid_added": False, "reason": "mid too close (index)"}

    sigma_mid = float(sigmas[k_mid])
    for k in idx:
        sigma_k = float(sigmas[k])
        if abs(np.log(sigma_mid / sigma_k)) < log_sigma_tol:
            return idx, {"mid_added": False, "reason": "mid too close (log-sigma)"}

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
        }

    return idx, {"mid_added": False, "reason": "redundant after pruning"}


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
):
    if len(sigmas) < 10 or len(score) < 10 or ls_eq is None or len(modes) == 0:
        return mode_sigmas, modes, {
            "added_terminal": False,
            "reason": "insufficient data",
        }

    sigmas = np.asarray(sigmas, float)
    score = np.asarray(score, float)

    j0 = max(int((1 - tail_frac) * len(score)), 0)
    tail = score[j0:]
    if len(tail) < 5:
        return mode_sigmas, modes, {"added_terminal": False, "reason": "tail too short"}

    j_peak = j0 + int(np.argmax(tail))

    prom = peak_prominence(score, j_peak, left_right=12)
    prom_global = float(
        np.max([peak_prominence(score, j, left_right=12) for j in range(1, len(score) - 1)])
    )
    if prom_global <= 0:
        return mode_sigmas, modes, {
            "added_terminal": False,
            "reason": "no global prominence",
        }

    if prom < min_prom_frac * prom_global:
        return mode_sigmas, modes, {
            "added_terminal": False,
            "reason": f"tail peak not prominent enough ({prom:.4g})",
        }

    k = j_peak + 1
    sigma_terminal = float(sigmas[k])
    candidate = reps[k]

    last_mode = modes[-1]
    sinu_last = global_sinuosity(last_mode)
    sinu_candidate = global_sinuosity(candidate)

    dist_last = mean_distance_to_original(ls_eq, last_mode, n=600)
    dist_candidate = mean_distance_to_original(ls_eq, candidate, n=600)

    if (sinu_last - sinu_candidate) < min_sinu_drop:
        return mode_sigmas, modes, {
            "added_terminal": False,
            "reason": "no extra sinuosity drop",
        }

    if dist_candidate < (1 + min_dist_increase) * max(dist_last, 1e-9):
        return mode_sigmas, modes, {
            "added_terminal": False,
            "reason": "no extra deviation increase",
        }

    mode_sigmas2 = np.append(mode_sigmas, sigma_terminal)
    modes2 = list(modes) + [candidate]
    return mode_sigmas2, modes2, {
        "added_terminal": True,
        "sigma": sigma_terminal,
        "prominence": float(prom),
    }


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
