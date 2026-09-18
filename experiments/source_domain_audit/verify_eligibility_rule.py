"""Programmatic verification of the manuscript's target-unit eligibility rule.

The manuscript (Data and target units) defines three eligibility criteria:
  1. the local GLORIA release codes the Site_name as a lake;
  2. at least 20 canonical groups;
  3. at least 3 sampling dates OR at least 2 Dataset_ID values.

gloria.eligible_tss_lakes() implements criteria 1-2; the frozen pipeline
scripts encode criterion 3 as the explicit exclusion of Ba Be Lake and
Lake Constance (STRICT_24_EXCLUDE). This script recomputes criterion 3
directly from the raw metadata on the frozen canonical-group population
and asserts that the programmatic rule and the frozen exclusion set
describe exactly the same 26 -> 24 outcome, so the hard-coded names act
as a regression anchor rather than as the definition.

Run:  python experiments/source_domain_audit/verify_eligibility_rule.py
Requires data/raw GLORIA metadata (see data/raw/DATA_SOURCES.md).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lake_insitu import gloria  # noqa: E402

MIN_GROUP = 20
FROZEN_EXCLUDE = {"Ba Be Lake", "Lake Constance"}
# The 24 eligible target units, from Supplementary Table S2 Panel A.
TABLE_S2_TARGETS = {
    "Branched Oak Lake", "Chaohu", "Dianchi", "Eagle Creek Reservoir",
    "Erhai", "Garda", "Geist Reservoir", "High Rock Lake",
    "Ibitinga Reservoir", "Lake Chotkowskie", "Lake Erie", "Lake Geneva",
    "Lake Hume", "Lake Jasień Południowy", "Lake Jeleń",
    "Lake Kasumigaura", "Lake Kummerow", "Lake Obłęże", "Lake Peipsi",
    "Lake Winnipeg", "Lake Łebsko", "Mantova", "Morse Reservoir", "Taihu",
}


def main():
    meta = gloria.load_meta()
    rrs = gloria.load_rrs()
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols)
    valid_keys = set(groups["group_id"])
    print(f"canonical groups (all water bodies): {len(valid_keys)}")
    assert len(valid_keys) == 4292

    # Re-derive each raw row's canonical key (same formula as
    # gloria.build_tss_groups) and keep rows whose group survived the
    # band-completeness drop, so date/Dataset_ID counts live on exactly
    # the frozen group population.
    tss = meta[meta["TSS"].notna()].copy()
    tss["date_day"] = tss["Date_Time_UTC"].dt.floor("D")
    tss["group_key"] = (
        tss["Site_name"].astype(str) + "||" +
        tss["Country"].astype(str) + "||" +
        tss["date_day"].astype(str) + "||" +
        tss["Latitude"].round(3).astype(str) + "||" +
        tss["Longitude"].round(3).astype(str)
    )
    tss = tss.merge(rrs[["GLORIA_ID"]], on="GLORIA_ID", how="inner")
    tss = tss[tss["group_key"].isin(valid_keys)]

    lake_rows = tss[tss["Water_body_type"] == gloria.LAKE_RESERVOIR_TYPE]
    per_site = lake_rows.groupby("Site_name").agg(
        n_groups=("group_key", "nunique"),
        n_dates=("date_day", lambda s: s.dropna().nunique()),
        n_dataset_ids=("Dataset_ID", "nunique"),
    )
    size_eligible = per_site[per_site.n_groups >= MIN_GROUP]
    criterion3 = (size_eligible.n_dates >= 3) | (size_eligible.n_dataset_ids >= 2)
    eligible = size_eligible[criterion3]

    failed = set(size_eligible.index) - set(eligible.index)
    eligible_groups = int(
        groups[(groups.Water_body_type == gloria.LAKE_RESERVOIR_TYPE)
               & groups.Site_name.isin(eligible.index)]["group_id"].nunique())

    print(f"size-eligible lake identifiers (>= {MIN_GROUP} groups): "
          f"{len(size_eligible)}")
    print(f"fail criterion 3 (dates<3 AND dataset_ids<2): {sorted(failed)}")
    print(f"eligible target units: {len(eligible)}")
    print(f"eligible target groups: {eligible_groups}")

    assert failed == FROZEN_EXCLUDE, (
        f"programmatic rule excludes {sorted(failed)}, frozen pipeline "
        f"excludes {sorted(FROZEN_EXCLUDE)} -- the two definitions have "
        f"diverged and the manuscript text or the pipeline must be updated")
    assert set(eligible.index) == TABLE_S2_TARGETS, (
        "eligible set no longer matches Supplementary Table S2 Panel A")
    assert len(eligible) == 24 and eligible_groups == 1755

    print("\nALL ASSERTIONS PASSED: the programmatic three-criterion rule "
          "reproduces the frozen 26 -> 24 eligibility outcome exactly.")


if __name__ == "__main__":
    main()
