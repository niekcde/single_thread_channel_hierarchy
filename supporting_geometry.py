"""Geometry helpers for the river hierarchy mode pipeline.

This module is intentionally flat so it can be imported directly from notebooks:

    from supporting_geometry import merge_paths_mainstem_only

It can also be run directly for a tiny smoke demo:

    python supporting_geometry.py
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point


def merge_paths_mainstem_only(gdf):
    """Merge the geometries in the supplied GeoDataFrame into one line.

    The caller is responsible for passing only the reaches/path segments that
    should be included. This function does not filter mainstem edges itself.
    """
    import duckdb
    from shapely import wkt

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    if gdf.empty:
        raise ValueError("Cannot merge paths from an empty GeoDataFrame")

    df = gdf.copy(deep=True)
    df = df[df["geometry"].notna() & ~df["geometry"].is_empty]
    if df.empty:
        raise ValueError("Cannot merge paths without valid geometries")

    df["geom_wkt"] = df["geometry"].apply(lambda g: g.wkt)
    df = df[["geom_wkt"]]

    con.register("segments", df)
    merged = con.execute(
        """
        SELECT
            ST_AsText(
                ST_LineMerge(
                    ST_Collect(
                        LIST(ST_GeomFromText(geom_wkt))
                    )
                )
            ) AS merged_wkt
        FROM segments
        """
    ).fetchdf()

    return wkt.loads(merged.loc[0, "merged_wkt"])


def resample_linestring_equal(ls: LineString, step: float) -> LineString:
    step = float(step)
    length = float(ls.length)
    if length <= 0 or step <= 0:
        return ls

    n = int(np.floor(length / step))
    dists = np.linspace(0, length, n + 2)
    pts = [ls.interpolate(d).coords[0] for d in dists]
    return LineString(pts)


def gaussian_kernel_1d(sigma_pts: float) -> np.ndarray:
    if sigma_pts <= 0:
        return np.array([1.0])

    radius = int(np.ceil(3 * sigma_pts))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma_pts**2))
    kernel /= kernel.sum()
    return kernel


def smooth_linestring_gaussian(
    ls: LineString,
    sigma_m: float,
    step_m: float,
) -> LineString:
    if sigma_m <= 0:
        return ls

    xy = np.asarray(ls.coords, float)
    if len(xy) < 3:
        return ls

    sigma_pts = float(sigma_m) / float(step_m)
    kernel = gaussian_kernel_1d(sigma_pts)
    pad = len(kernel) // 2

    x = np.pad(xy[:, 0], (pad, pad), mode="edge")
    y = np.pad(xy[:, 1], (pad, pad), mode="edge")

    xs = np.convolve(x, kernel, mode="valid")
    ys = np.convolve(y, kernel, mode="valid")

    out = np.column_stack([xs, ys])
    out[0] = xy[0]
    out[-1] = xy[-1]
    return LineString(out.tolist())


def simplify_mode(
    mode: LineString,
    sigma: float,
    *,
    eps_floor: float = 2.0,
    eps_factor: float = 1.0,
    power: float = 0.5,
) -> LineString:
    eps = float(eps_floor + eps_factor * (max(float(sigma), 0.0) ** float(power)))
    out = mode.simplify(eps, preserve_topology=False)
    if len(out.coords) < 2:
        out = LineString([mode.coords[0], mode.coords[-1]])
    return out


def snap_vertices_to_original(mode: LineString, original: LineString) -> LineString:
    pts = []
    for x, y in mode.coords:
        distance_along_original = original.project(Point(x, y))
        point = original.interpolate(distance_along_original)
        pts.append(point.coords[0])

    if len(pts) < 2:
        pts = [original.coords[0], original.coords[-1]]
    return LineString(pts)


def smooth_spline_linestring(
    ls: LineString,
    *,
    n: int = 400,
    smooth_factor: float = 0.001,
) -> LineString:
    """Parametric spline smoothing.

    ``smooth_factor`` is relative; smaller values stay closer to the original.
    SciPy is imported lazily so the main pipeline does not require it unless
    this function is used.
    """
    from scipy.interpolate import splprep, splev

    xy = np.asarray(ls.coords)
    if len(xy) < 4:
        return ls

    x, y = xy[:, 0], xy[:, 1]
    smoothing = smooth_factor * (ls.length**2)

    tck, _ = splprep([x, y], s=smoothing, k=min(3, len(xy) - 1))
    uu = np.linspace(0, 1, n)
    out = np.vstack(splev(uu, tck)).T
    return LineString(out)


def _demo() -> None:
    x = np.linspace(0, 1000, 80)
    y = 80 * np.sin(x / 90)
    line = LineString(np.column_stack([x, y]))

    equal = resample_linestring_equal(line, step=20)
    smooth = smooth_linestring_gaussian(equal, sigma_m=80, step_m=20)

    print("original points:", len(line.coords))
    print("resampled points:", len(equal.coords))
    print("smoothed length:", round(smooth.length, 2))


if __name__ == "__main__":
    _demo()
