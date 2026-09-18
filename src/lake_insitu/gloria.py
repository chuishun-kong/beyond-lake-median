"""GLORIA dataset loading, deduplication and group construction.

Group definition follows a validated duplicate finding:
(Site_name, Country, date floored to day) covers 89% of GLORIA as near-duplicates.
Active-selection candidate units must be these groups, not raw rows. The
canonical key is site/date/rounded-coordinate; campaign, provider, and
Dataset_ID are not key components.
"""
from pathlib import Path

import numpy as np
import pandas as pd

GLORIA_DIR = Path("data/raw/gloria/GLORIA_2022")
LAKE_RESERVOIR_TYPE = 1


def load_meta(gloria_dir: Path = GLORIA_DIR) -> pd.DataFrame:
    meta = pd.read_csv(gloria_dir / "GLORIA_meta_and_lab.csv", low_memory=False)
    meta["Date_Time_UTC"] = pd.to_datetime(meta["Date_Time_UTC"], errors="coerce")
    return meta


def load_rrs(gloria_dir: Path = GLORIA_DIR) -> pd.DataFrame:
    rrs = pd.read_csv(gloria_dir / "GLORIA_Rrs.csv", low_memory=False)
    return rrs


def rrs_band_columns(rrs: pd.DataFrame, lo: int = 400, hi: int = 800) -> list:
    """Restrict to a usable hyperspectral VIS-NIR window; GLORIA spans 350-900nm
    but edge bands are noisier / less populated across instruments.

    The frozen manuscript protocol is 400--750 nm: every experiment entry
    point passes lo=400, hi=750 explicitly. The hi=800 default predates the
    freeze and is kept only for historical pre-gate callers; do not rely on
    it for new analyses."""
    cols = []
    for c in rrs.columns:
        if not c.startswith("Rrs_"):
            continue
        wl = int(c.split("_")[1])
        if lo <= wl <= hi:
            cols.append(c)
    return cols


def build_tss_groups(meta: pd.DataFrame, rrs: pd.DataFrame,
                      band_cols: list, drop_incomplete_spectra: bool = True,
                      coord_precision: int = 3) -> pd.DataFrame:
    """Filter to TSS-labeled rows, join Rrs, and aggregate technical replicates
    within (Site_name, Country, coordinates, date-day) to one row per group
    via the median.

    coord_precision=3 (~111m at the equator) distinguishes distinct sampling
    STATIONS within the same waterbody+day from true repeat casts at the same
    spot. This matters a lot: grouping by (Site_name, Country, date-day) alone
    -- i.e. treating "same lake, same day" as one group regardless of location
    -- collapses genuinely distinct spatial survey stations into one median,
    destroying real within-lake spatial/optical heterogeneity. Verified
    concretely: Taihu 2008-10-14 has 20 raw rows spanning TSS 14-246 mg/L
    (std=52) across a >10km transect on a single day; grouping by lake+day
    alone would average these into one synthetic point. Adding coordinates
    recovers 27 eligible (n_group>=20) lakes/reservoirs vs only 6 without
    them. The canonical key keeps site, date, and rounded coordinates distinct; it does
    not include campaign, provider, or Dataset_ID as a grouping component.

    Returns one row per group with columns: group_id, Site_name, Country,
    Water_body_type, date_day, TSS (group median), and the Rrs band columns
    (group median spectrum). Groups with any NaN band in the requested range
    (a handful of source-only waterbodies with narrower instrument coverage)
    are dropped by default -- verified to never affect the eligible lakes at
    the default 400-750nm range.
    """
    tss = meta[meta["TSS"].notna()].copy()
    tss["date_day"] = tss["Date_Time_UTC"].dt.floor("D")
    tss["group_key"] = (
        tss["Site_name"].astype(str) + "||" +
        tss["Country"].astype(str) + "||" +
        tss["date_day"].astype(str) + "||" +
        tss["Latitude"].round(coord_precision).astype(str) + "||" +
        tss["Longitude"].round(coord_precision).astype(str)
    )

    joined = tss.merge(rrs[["GLORIA_ID"] + band_cols], on="GLORIA_ID", how="inner")

    non_spectral = joined.groupby("group_key").agg(
        Site_name=("Site_name", "first"),
        Country=("Country", "first"),
        Water_body_type=("Water_body_type", lambda s: s.mode().iloc[0]),
        date_day=("date_day", "first"),
        n_replicates=("GLORIA_ID", "count"),
        TSS=("TSS", "median"),
    )
    spectral = joined.groupby("group_key")[band_cols].median()
    agg = pd.concat([non_spectral, spectral], axis=1)

    agg = agg.reset_index().rename(columns={"group_key": "group_id"})
    if drop_incomplete_spectra:
        agg = agg[~agg[band_cols].isna().any(axis=1)].reset_index(drop=True)
    return agg


def eligible_tss_lakes(groups: pd.DataFrame, min_group: int = 20,
                        lakes_reservoirs_only: bool = True) -> pd.Series:
    """Return Site_name -> n_group counts for lakes meeting the group-count
    threshold (eligible-lake definition, n_group >= 20 used
    for the pre-gate; full MVP additionally checks date/campaign diversity).

    NOTE: this helper implements only criteria 1 (lake-labelled) and 2
    (>= min_group canonical groups) of the manuscript's three target-unit
    eligibility rules. The third rule -- at least 3 sampling dates OR at
    least 2 Dataset_ID values -- is verified programmatically by
    experiments/source_domain_audit/verify_eligibility_rule.py, which
    reproduces the 26 size-eligible -> 24 eligible outcome (Ba Be Lake and
    Lake Constance fail the diversity rule) that the frozen pipeline
    encodes via explicit exclusion sets."""
    df = groups
    if lakes_reservoirs_only:
        df = df[df["Water_body_type"] == LAKE_RESERVOIR_TYPE]
    counts = df.groupby("Site_name")["group_id"].nunique().sort_values(ascending=False)
    return counts[counts >= min_group]


def split_outer_target_source(groups: pd.DataFrame, target_site: str,
                              additional_source_exclusions=None):
    """Split one outer target from its source archive by exact Site_name.

    ``additional_source_exclusions`` supports a declared identity-boundary
    sensitivity without changing which rows form the outer target.  The
    default reproduces the frozen all-other-GLORIA source regime.
    """
    extra = set(additional_source_exclusions or ())
    target = groups[groups["Site_name"] == target_site].reset_index(drop=True)
    excluded_from_source = extra | {target_site}
    source = groups[~groups["Site_name"].isin(excluded_from_source)].reset_index(drop=True)
    return target, source
