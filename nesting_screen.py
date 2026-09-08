#!/usr/bin/env python
"""
nesting_screen.py

Spatial-independence screen for the analysis set.

The 2,135 catchments are NOT 2,135 independent draws. Their areas sum to about
6.7 million km^2 against roughly 2.8 million km^2 of land in the 16 countries, so the
same water is gauged repeatedly: LamaH-CE and GRDC-Caravan both carry long gauge
chains down a main stem. The four largest catchments in the sample are four gauges
on the Rhine (158.7k, 147.5k, 144.0k and 139.2k km^2) whose memory timescales agree to
within a few days. Counting them as four observations inflates the effective sample
size, deflates every standard error, and lets a nested cluster sit wholly inside one
country's leave-one-country-out holdout.

Caravan carries no catchment topology (no NEXT_DOWN, no upstream-gauge list) and
polygons are available for only one source, so nesting is inferred geometrically.
Treating each catchment as a disc of its own area, gauge B is taken to lie inside
catchment A when the great-circle distance between the gauges is under A's equivalent
radius sqrt(area_A / pi) and area_B <= area_A. This is a PROXY: it over-flags where two
separate basins sit side by side inside a large one's radius, which is the conservative
direction for a sensitivity analysis.

Two products are written:

  is_container   catchment holds at least one retained smaller gauge -- the flag used
                 to build the independent subset. The screen is greedy from the
                 SMALLEST catchment up, so headwaters are kept and the large nesting
                 containers are dropped: headwaters are the independent units, and a
                 lumped conceptual model is on firmer ground at 200 km^2 than at
                 150,000 km^2.
  same_stem_pair descriptive only: pairs that are almost certainly consecutive gauges
                 on one main stem (distance under the larger equivalent radius AND
                 area ratio >= 0.5), reported to quantify the duplication directly.

    python nesting_screen.py --strength-memory <response_strength_memory_DJF.parquet> --refined <..._refined.csv> \
        --out <nesting_flags.csv>
    python nesting_screen.py --selftest
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

R_EARTH = 6371.0088          # km
STEM_AREA_RATIO = 0.5        # area_small / area_large above this = same main stem


def haversine_matrix(lat, lon):
    """Pairwise great-circle distance (km) between gauges."""
    la = np.radians(np.asarray(lat, dtype=float))
    lo = np.radians(np.asarray(lon, dtype=float))
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def equivalent_radius(area_km2):
    return np.sqrt(np.asarray(area_km2, dtype=float) / np.pi)


def screen(df: pd.DataFrame) -> pd.DataFrame:
    """Greedy smallest-first independence screen.

    Walk the catchments from smallest to largest. A candidate is dropped as a
    CONTAINER when a gauge already retained falls within the candidate's equivalent
    radius: that retained catchment is (proxy-)nested inside it, so keeping both
    double-counts the same water. Requires columns gauge_id, gauge_lat, gauge_lon, area.
    """
    d = df.reset_index(drop=True)
    order = np.argsort(d["area"].to_numpy())          # smallest first
    dist = haversine_matrix(d["gauge_lat"], d["gauge_lon"])
    rad = equivalent_radius(d["area"])

    retained: list[int] = []
    is_container = np.zeros(len(d), dtype=bool)
    n_contained = np.zeros(len(d), dtype=int)
    for i in order:
        if retained:
            inside = dist[i, retained] < rad[i]
            n_contained[i] = int(inside.sum())
            if inside.any():
                is_container[i] = True
                continue
        retained.append(i)

    out = d[["gauge_id"]].copy()
    out["area"] = d["area"]
    out["equiv_radius_km"] = rad
    out["n_gauges_inside"] = n_contained
    out["is_container"] = is_container
    out["independent"] = ~is_container
    return out


def same_stem_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs that are almost certainly consecutive gauges on one main stem."""
    d = df.reset_index(drop=True)
    dist = haversine_matrix(d["gauge_lat"], d["gauge_lon"])
    area = d["area"].to_numpy(dtype=float)
    rad = equivalent_radius(area)
    big = np.maximum.outer(area, area)
    small = np.minimum.outer(area, area)
    rad_big = equivalent_radius(big)
    iu = np.triu_indices(len(d), k=1)
    ratio = small[iu] / np.maximum(big[iu], 1e-9)
    sel = (dist[iu] < rad_big[iu]) & (ratio >= STEM_AREA_RATIO)
    i, j = iu[0][sel], iu[1][sel]
    return pd.DataFrame({
        "gauge_a": d.gauge_id.to_numpy()[i], "gauge_b": d.gauge_id.to_numpy()[j],
        "area_a": area[i], "area_b": area[j],
        "dist_km": dist[iu][sel], "area_ratio": ratio[sel],
    }).sort_values("area_a", ascending=False)


def validate(flags: pd.DataFrame, d: pd.DataFrame, pairs: pd.DataFrame):
    print("=== spatial-independence screen ===", flush=True)
    n, k = len(flags), int(flags.independent.sum())
    print(f"catchments: {n}   independent: {k}   dropped as containers: {n - k}",
          flush=True)
    a_all = d["area"].sum()
    a_ind = d.loc[flags.independent.to_numpy(), "area"].sum()
    print(f"summed area: all {a_all/1e6:.2f}e6 km^2  ->  independent {a_ind/1e6:.2f}e6 km^2",
          flush=True)
    print(f"  (land area of the 16 countries is about 2.8e6 km^2)", flush=True)
    med = d["area"].median(), d.loc[flags.independent.to_numpy(), "area"].median()
    print(f"median area: all {med[0]:.0f} km^2  ->  independent {med[1]:.0f} km^2", flush=True)
    print(f"\nsame-stem pairs (distance < equivalent radius, area ratio >= "
          f"{STEM_AREA_RATIO}): {len(pairs)}", flush=True)
    print("largest such pairs:", flush=True)
    print(pairs.head(6).to_string(index=False), flush=True)
    if "country" in d.columns:
        t = pd.DataFrame({"country": d["country"].to_numpy(),
                          "ind": flags.independent.to_numpy()})
        g = t.groupby("country")["ind"].agg(["size", "sum"])
        g.columns = ["n_all", "n_independent"]
        print("\nby country:", flush=True)
        print(g.sort_values("n_all", ascending=False).to_string(), flush=True)


def selftest():
    # A: a large container at the origin; B, C: two small gauges well inside it;
    # D: a small gauge far outside. Expect A dropped, B/C/D retained.
    d = pd.DataFrame({
        "gauge_id": ["A", "B", "C", "D"],
        "gauge_lat": [50.0, 50.2, 49.8, 60.0],
        "gauge_lon": [8.0, 8.1, 7.9, 8.0],
        "area": [100000.0, 300.0, 250.0, 400.0],
    })
    f = screen(d).set_index("gauge_id")
    assert bool(f.loc["A", "is_container"]), f
    assert f.loc[["B", "C", "D"], "independent"].all(), f
    assert int(f.loc["A", "n_gauges_inside"]) == 2, f

    # two same-size neighbours on one stem: the screen keeps exactly one
    d2 = pd.DataFrame({
        "gauge_id": ["P", "Q"],
        "gauge_lat": [50.0, 50.3], "gauge_lon": [8.0, 8.0],
        "area": [20000.0, 19000.0],
    })
    f2 = screen(d2)
    assert int(f2.independent.sum()) == 1, f2
    p2 = same_stem_pairs(d2)
    assert len(p2) == 1 and p2.area_ratio.iloc[0] > 0.9, p2

    # far-apart equals are both independent and are not a stem pair
    d3 = pd.DataFrame({
        "gauge_id": ["X", "Y"], "gauge_lat": [45.0, 60.0], "gauge_lon": [0.0, 20.0],
        "area": [500.0, 500.0],
    })
    assert int(screen(d3).independent.sum()) == 2
    assert len(same_stem_pairs(d3)) == 0

    # haversine sanity: one degree of latitude is about 111 km
    assert abs(haversine_matrix([0.0, 1.0], [0.0, 0.0])[0, 1] - 111.19) < 0.5
    print("selftest OK  (container dropped, headwaters kept, stem pair detected)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength-memory")
    ap.add_argument("--refined")
    ap.add_argument("--out")
    ap.add_argument("--pairs-out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not all([a.strength_memory, a.out]):
        ap.error("--strength-memory and --out required unless --selftest")
    d = pd.read_parquet(a.strength_memory)[["gauge_id", "gauge_lat", "gauge_lon", "area", "source"]]
    if a.refined:
        country = pd.read_csv(a.refined).set_index("gauge_id")["country"]
        d["country"] = d.gauge_id.map(country)
        gb = {"England", "Scotland", "Wales", "Great Britain"}
        d["country"] = d["country"].where(~d["country"].isin(gb), "Great Britain")
    d = d.dropna(subset=["gauge_lat", "gauge_lon", "area"]).reset_index(drop=True)
    flags = screen(d)
    pairs = same_stem_pairs(d)
    flags.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(flags)} catchments)", flush=True)
    if a.pairs_out:
        pairs.to_csv(a.pairs_out, index=False)
        print(f"wrote {a.pairs_out}  ({len(pairs)} pairs)", flush=True)
    validate(flags, d, pairs)


if __name__ == "__main__":
    main()
