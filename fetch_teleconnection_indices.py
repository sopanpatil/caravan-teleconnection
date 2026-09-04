"""
fetch_teleconnection_indices.py

Downloads North Atlantic teleconnection indices and builds a tidy seasonal table
keyed by winter-year, ready to join to catchment / HBV output.

Source:
- NAO, EA, EA/WR, SCA : NOAA CPC, monthly 1950-, standardised to 1981-2010.
  https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/tele_index.nh

Conventions baked in:
- DJF winter labelled by its January's year: Dec(Y-1)+Jan(Y)+Feb(Y) -> winter_year = Y.
- A season is NaN unless every month is present.
"""
from __future__ import annotations
import re
import urllib.request
import pandas as pd

CPC_URL = "https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/tele_index.nh"
SENTINEL = -99.0  # anything <= this is missing in both files

CPC_COLS = ["NAO", "EA", "WP", "EPNP", "PNA", "EAWR", "SCA", "TNH", "POL", "PT", "ExplVar"]
KEEP = ["NAO", "EA", "EAWR", "SCA"]


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def _dejam(line: str) -> str:
    """Separate fixed-width numeric fields that run together.

    These files are fixed-width, so a negative value (e.g. the -99.90 sentinel)
    following another negative value has no separating space: '-2.16-99.90'.
    A plain str.split() then yields too few tokens and the whole row is dropped.
    Insert a space before any '-' that directly follows a digit; a minus can
    only sit mid-token as the sign of a jammed next field (no sci-notation here).
    """
    return re.sub(r"(?<=[0-9])-", " -", line)


def load_cpc(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        t = _dejam(line).split()
        if len(t) >= 13 and t[0].isdigit() and t[1].isdigit() and 1 <= int(t[1]) <= 12:
            rows.append([int(t[0]), int(t[1])] + [float(x) for x in t[2:13]])
    df = pd.DataFrame(rows, columns=["year", "month"] + CPC_COLS)
    df[CPC_COLS] = df[CPC_COLS].where(df[CPC_COLS] > SENTINEL)
    return df[["year", "month"] + KEEP]


def seasonal(monthly, value_cols, months, rollover, suffix):
    d = monthly[monthly["month"].isin(months)].copy()
    d["winter_year"] = d["year"] + d["month"].isin(rollover).astype(int)
    g = d.groupby("winter_year")
    out = g[value_cols].mean().where(g[value_cols].count().eq(len(months)))
    return out.add_suffix(suffix)


def build_table() -> pd.DataFrame:
    monthly = load_cpc(_get(CPC_URL)).sort_values(["year", "month"])
    table = seasonal(monthly, KEEP, [12, 1, 2], {12}, "_DJF").sort_index()
    table.index.name = "winter_year"
    return table


if __name__ == "__main__":
    tbl = build_table()
    tbl.to_csv("teleconnection_seasonal.csv")
    print(f"winter_year range: {tbl.index.min()}-{tbl.index.max()}, rows: {len(tbl)}")
    print("columns:", list(tbl.columns))
    print(tbl.loc[[1989, 1990, 2010], ["NAO_DJF", "EA_DJF", "EAWR_DJF", "SCA_DJF"]])
