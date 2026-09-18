"""China lakes (Zhai et al. 2024) loading and grouping -- replication/
stress-test role only (this dataset is within FSResTL-Chla's IWC target
domain, so it can never be claimed as independent external validation).

Same coordinate+date-aware grouping as gloria.py, adapted to this dataset's
column schema (Lake name / TSM(mg/L) / plain-digit wavelength columns).
"""
from pathlib import Path

import numpy as np
import pandas as pd

CHINA_DIR = Path("data/raw/china_lakes")

IWC_ONLY_LAKES = [
    "Lake Khanka", "Lake Chagan", "Lake Hulun", "Lake Gaoyou", "Lake Bosten",
    "Lake Baiyangdian", "Lake Tuosu", "Lake Butbaruch", "Lake Qinghai",
    "Lake Ballet Swan", "Lake Changdang", "Lake Ge", "Lake Hong-tse",
]  # per data/processed/overlap_audit_china.csv: zero overlap with GLORIA


def load_merged(china_dir: Path = CHINA_DIR) -> pd.DataFrame:
    meta = pd.read_csv(china_dir / "Meta.csv", low_memory=False)
    wp = pd.read_csv(china_dir / "water_parameter.csv", low_memory=False)
    rrs = pd.read_csv(china_dir / "remote_sensing_reflectance.csv", low_memory=False)
    merged = meta.merge(wp[["OID", "TSM(mg/L)"]], on="OID", how="left")
    merged = merged.merge(rrs.drop(columns=["Lake name", "Date", "Longitude", "Latitude"],
                                     errors="ignore"), on="OID", how="left")
    merged["Date_parsed"] = pd.to_datetime(
        merged["Date"].str.replace("_", "-", regex=False), errors="coerce")
    return merged


def band_columns(merged: pd.DataFrame, lo: int = 400, hi: int = 750) -> list:
    return [c for c in merged.columns if c.isdigit() and lo <= int(c) <= hi]


def build_tsm_groups(merged: pd.DataFrame, band_cols: list,
                      coord_precision: int = 3) -> pd.DataFrame:
    """Same group definition as gloria.build_tss_groups: (lake, coords
    rounded, date-day). Aggregates technical replicates via median."""
    df = merged[merged["TSM(mg/L)"].notna()].copy()
    df["date_day"] = df["Date_parsed"].dt.floor("D")
    df["group_key"] = (
        df["Lake name"].astype(str) + "||" +
        df["date_day"].astype(str) + "||" +
        df["Latitude"].round(coord_precision).astype(str) + "||" +
        df["Longitude"].round(coord_precision).astype(str)
    )
    non_spectral = df.groupby("group_key").agg(
        Lake_name=("Lake name", "first"),
        date_day=("date_day", "first"),
        n_replicates=("OID", "count"),
        TSM=("TSM(mg/L)", "median"),
    )
    spectral = df.groupby("group_key")[band_cols].median()
    agg = pd.concat([non_spectral, spectral], axis=1).reset_index()
    agg = agg.rename(columns={"group_key": "group_id"})
    agg = agg[~agg[band_cols].isna().any(axis=1)].reset_index(drop=True)
    return agg
