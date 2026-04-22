"""Metric helpers for river centerline scale-space modes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from shapely.geometry import LineString


def global_sinuosity(ls: LineString) -> float:
    xy = np.asarray(ls.coords, float)
    if len(xy) < 2:
        return 1.0

    chord = np.linalg.norm(xy[-1] - xy[0])
    return float(ls.length / max(chord, 1e-12))


def turning_energy(ls: LineString) -> float:
    """Curvature proxy: sum of absolute turn angles divided by length."""
    xy = np.asarray(ls.coords, float)
    if len(xy) < 3 or ls.length <= 0:
        return 0.0

    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    mask = (n1 > 1e-12) & (n2 > 1e-12)

    v1 = v1[mask] / n1[mask, None]
    v2 = v2[mask] / n2[mask, None]

    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = np.sum(v1 * v2, axis=1)
    angles = np.abs(np.arctan2(cross, dot))
    return float(np.sum(angles) / ls.length)


def mean_distance_to_original(
    original: LineString,
    other: LineString,
    n: int = 600,
) -> float:
    length = float(original.length)
    if length <= 0:
        return 0.0

    dists = np.linspace(0, length, n)
    pts = [original.interpolate(d) for d in dists]
    return float(np.mean([other.distance(point) for point in pts]))


def curvature_sign_changes(ls: LineString) -> int:
    xy = np.asarray(ls.coords, float)
    if len(xy) < 3:
        return 0

    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(cross)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def curvature_lobes(ls: LineString) -> int:
    sign_changes = curvature_sign_changes(ls)
    return int(sign_changes // 2 + 1)


def classify_modes_by_extrema(modes: Sequence[LineString]):
    sign_changes = np.array([curvature_sign_changes(mode) for mode in modes], int)
    lobes = np.array([curvature_lobes(mode) for mode in modes], int)

    if len(sign_changes) >= 3:
        q1, q2 = np.quantile(sign_changes, [0.33, 0.66])
    else:
        q1, q2 = np.min(sign_changes), np.max(sign_changes)

    labels = []
    for sign_change_count in sign_changes:
        if sign_change_count <= max(2, q1):
            labels.append("reach/valley-axis")
        elif sign_change_count <= q2:
            labels.append("macro / compound-bend")
        else:
            labels.append("meander / high-complexity")

    return sign_changes, lobes, labels


def _demo() -> None:
    x = np.linspace(0, 1000, 100)
    y = 60 * np.sin(x / 80)
    line = LineString(np.column_stack([x, y]))

    print("sinuosity:", round(global_sinuosity(line), 3))
    print("turning energy:", round(turning_energy(line), 6))
    print("curvature sign changes:", curvature_sign_changes(line))


if __name__ == "__main__":
    _demo()
