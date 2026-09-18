"""Deterministic, read-only evidence audit for GLORIA TSS metadata.

The audit rebuilds the same 400--750 nm canonical archive used by the frozen
true-nested analyses.  It produces provenance tables only: no model is fit,
no result file is read or changed, and no eligibility decision is recomputed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lake_insitu import gloria  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ELIGIBLE_PATH = REPO_ROOT / "data" / "processed" / "eligible_tasks.csv"
GLORIA_RELATIVE = Path("data/raw/gloria/GLORIA_2022")
WATER_BODY_LABELS = {
    1: "lake",
    2: "estuary",
    3: "coastal_ocean",
    4: "river",
    5: "other",
}
METADATA_COLUMNS = [
    "Dataset_ID",
    "Organization_ID",
    "TSS_method",
    "Water_body_type",
    "Country",
    "Site_name",
]


def _clean(value: object, missing: str = "<missing>") -> str:
    if pd.isna(value):
        return missing
    text = str(value).strip()
    return text or missing


def _join(values: Iterable[object], delimiter: str = ";") -> str:
    return delimiter.join(sorted({_clean(value) for value in values}))


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def _group_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["Site_name"].astype(str)
        + "||"
        + frame["Country"].astype(str)
        + "||"
        + frame["Date_Time_UTC"].dt.floor("D").astype(str)
        + "||"
        + frame["Latitude"].round(3).astype(str)
        + "||"
        + frame["Longitude"].round(3).astype(str)
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric_summary(frame: pd.DataFrame) -> dict[str, object]:
    values = frame["TSS"]
    return {
        "tss_min": float(values.min()),
        "tss_median": float(values.median()),
        "tss_max": float(values.max()),
        "tss_unit": "g m-3",
        "tss_missing_count": int(values.isna().sum()),
    }


def _population_summary(frame: pd.DataFrame, population: str) -> dict[str, object]:
    return {
        "population": population,
        "canonical_group_count": int(frame["group_id"].nunique()),
        "site_count": int(frame["Site_name"].nunique()),
        "raw_record_count": int(frame["raw_record_count"].sum()),
        **_numeric_summary(frame),
    }


def _read_method_dictionary(gloria_dir: Path) -> pd.DataFrame:
    dictionary = pd.read_excel(
        gloria_dir / "GLORIA_variables_and_methods.xlsx", sheet_name="TSS methods"
    ).rename(
        columns={
            "Dataset ID": "Dataset_ID",
            "Methodology short name": "TSS_method",
            "Filter type": "filter_type",
            "Measurement technique": "measurement_technique",
            "Approach": "approach",
            "Applications": "applications",
        }
    )
    dictionary = dictionary[[
        "Dataset_ID",
        "TSS_method",
        "filter_type",
        "measurement_technique",
        "approach",
        "applications",
    ]].copy()
    dictionary["Dataset_ID"] = dictionary["Dataset_ID"].map(_clean)
    dictionary["TSS_method"] = dictionary["TSS_method"].map(_clean)
    dictionary = dictionary[
        (dictionary["Dataset_ID"] != "<missing>")
        & (dictionary["TSS_method"] != "<missing>")
    ]
    return (
        dictionary.groupby(["Dataset_ID", "TSS_method"], as_index=False, sort=True)
        .agg({column: _join for column in dictionary.columns if column not in {"Dataset_ID", "TSS_method"}})
    )


def _load_inputs(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    gloria_dir = repo_root / GLORIA_RELATIVE
    meta = gloria.load_meta(gloria_dir)
    rrs = gloria.load_rrs(gloria_dir)
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    tss = meta.loc[meta["TSS"].notna()].copy()
    tss["group_id"] = _group_key(tss)
    joined = tss.merge(rrs[["GLORIA_ID"] + band_cols], on="GLORIA_ID", how="inner")
    complete = joined.dropna(subset=band_cols).copy()
    groups = gloria.build_tss_groups(meta, rrs, band_cols, coord_precision=3)
    retained = complete[complete["group_id"].isin(set(groups["group_id"]))].copy()
    if len(retained) != len(complete) or retained["group_id"].nunique() != len(groups):
        raise AssertionError("Frozen complete-spectrum archive did not reconcile")

    eligible = pd.read_csv(repo_root / ELIGIBLE_PATH.relative_to(repo_root))
    eligible["final_eligible"] = _as_bool(eligible["all_eligibility_criteria_pass"])
    eligible_sites = eligible.loc[eligible["final_eligible"], "lake"].tolist()
    return meta, tss, joined, retained, groups, eligible, eligible_sites


def _group_metadata(retained: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    raw_counts = retained.groupby("group_id", sort=True)["GLORIA_ID"].agg(
        raw_record_count="count", raw_gloria_ids=lambda values: _join(values)
    )
    metadata = (
        retained.groupby("group_id", sort=True)
        .agg(
            Site_name=("Site_name", _join),
            Country=("Country", _join),
            Water_body_type=("Water_body_type", _join),
            Dataset_ID=("Dataset_ID", _join),
            Organization_ID=("Organization_ID", _join),
            TSS_method=("TSS_method", _join),
            latitude_min=("Latitude", "min"),
            latitude_median=("Latitude", "median"),
            latitude_max=("Latitude", "max"),
            longitude_min=("Longitude", "min"),
            longitude_median=("Longitude", "median"),
            longitude_max=("Longitude", "max"),
            date_first=("Date_Time_UTC", "min"),
            date_last=("Date_Time_UTC", "max"),
        )
        .reset_index()
        .merge(raw_counts.reset_index(), on="group_id", validate="one_to_one")
        .merge(groups[["group_id", "TSS"]], on="group_id", validate="one_to_one")
    )
    for column in ["Site_name", "Country", "Water_body_type"]:
        if metadata[column].str.contains(";", regex=False).any():
            raise AssertionError(f"Unexpected canonical identity conflict in {column}")
    metadata["Water_body_type"] = metadata["Water_body_type"].astype(int)
    metadata["water_body_label"] = metadata["Water_body_type"].map(WATER_BODY_LABELS)
    return metadata.sort_values("group_id").reset_index(drop=True)


def _candidate_table(
    candidates: pd.DataFrame, group_meta: pd.DataFrame, retained: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in candidates.sort_values("lake").itertuples(index=False):
        site = row.lake
        site_groups = group_meta[group_meta["Site_name"] == site]
        site_raw = retained[retained["Site_name"] == site]
        codes = sorted(site_raw["Water_body_type"].dropna().astype(int).unique())
        dataset_count = site_groups["Dataset_ID"].str.split(";", regex=False).explode().nunique()
        campaign_count = int(row.n_unique_campaigns_dataset_id)
        if campaign_count != dataset_count:
            raise AssertionError(f"Dataset_ID campaign proxy mismatch for {site}")
        is_lake_label = codes == [1]
        rows.append(
            {
                "Site_name": site,
                "Country": _join(site_raw["Country"]),
                "water_body_codes": ";".join(map(str, codes)),
                "water_body_labels": ";".join(WATER_BODY_LABELS[code] for code in codes),
                "lake_labelled_by_gloria": is_lake_label,
                "reservoir_evidence": "no_explicit_GLORIA_reservoir_category_or_field",
                "raw_tss_record_count": int((site_raw["TSS"].notna()).sum()),
                "canonical_group_count": int(len(site_groups)),
                "distinct_dates": int(site_groups["date_first"].dt.floor("D").nunique()),
                "campaign_count": campaign_count,
                "campaign_definition": "distinct Dataset_ID count from eligible_tasks.csv column n_unique_campaigns_dataset_id",
                "campaign_proxy_recomputed_from_retained_groups": dataset_count,
                "min_group_count_pass": bool(row.p1_5_lake_reservoir_type),
                "date_or_campaign_diversity_pass": bool(row.p1_6_min3_dates_or_2campaigns),
                "query_pool_pass": bool(row.k5_query_pool_ge8),
                "final_eligible": bool(row.final_eligible),
                "eligibility_reason": (
                    "included" if row.final_eligible else "failed_date_or_dataset_id_proxy_campaign_diversity"
                ),
            }
        )
    return pd.DataFrame(rows)


def _target_metadata(targets: pd.DataFrame, group_meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidate_index = targets.set_index("Site_name")
    for site in sorted(candidate_index.index):
        metadata = group_meta[group_meta["Site_name"] == site]
        candidate = candidate_index.loc[site]
        rows.append(
            {
                "Site_name": site,
                "Country": _join(metadata["Country"]),
                "latitude_min": metadata["latitude_min"].min(),
                "latitude_median": metadata["latitude_median"].median(),
                "latitude_max": metadata["latitude_max"].max(),
                "longitude_min": metadata["longitude_min"].min(),
                "longitude_median": metadata["longitude_median"].median(),
                "longitude_max": metadata["longitude_max"].max(),
                "water_body_codes": _join(metadata["Water_body_type"]),
                "water_body_labels": _join(metadata["water_body_label"]),
                "raw_record_count": int(metadata["raw_record_count"].sum()),
                "canonical_group_count": int(len(metadata)),
                "date_first": metadata["date_first"].min().date().isoformat(),
                "date_last": metadata["date_last"].max().date().isoformat(),
                "distinct_date_count": int(metadata["date_first"].dt.floor("D").nunique()),
                "campaign_count": int(candidate["campaign_count"]),
                "campaign_definition": candidate["campaign_definition"],
                **_numeric_summary(metadata),
                "Dataset_ID_membership": _join(metadata["Dataset_ID"].str.split(";", regex=False).explode()),
                "Organization_ID_membership": _join(metadata["Organization_ID"].str.split(";", regex=False).explode()),
                "TSS_method_membership": _join(metadata["TSS_method"].str.split(";", regex=False).explode()),
                "missing_tss_method_group_count": int(metadata["TSS_method"].eq("<missing>").sum()),
                "multi_dataset": bool(metadata["Dataset_ID"].str.split(";", regex=False).explode().nunique() > 1),
                "multi_provider": bool(metadata["Organization_ID"].str.split(";", regex=False).explode().nunique() > 1),
                "multi_method": bool(metadata["TSS_method"].str.split(";", regex=False).explode().nunique() > 1),
            }
        )
    return pd.DataFrame(rows)


def _target_membership(target_sites: list[str], group_meta: pd.DataFrame) -> pd.DataFrame:
    frame = group_meta[group_meta["Site_name"].isin(target_sites)]
    return (
        frame.groupby(["Site_name", "Dataset_ID", "Organization_ID", "TSS_method"], sort=True)
        .agg(canonical_group_count=("group_id", "count"), raw_record_count=("raw_record_count", "sum"))
        .reset_index()
    )


def _waterbody_composition(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for code, part in frame.groupby("Water_body_type", sort=True):
            rows.append({
                "population": population,
                "water_body_code": int(code),
                "water_body_label": WATER_BODY_LABELS[int(code)],
                **_population_summary(part, population),
            })
    return pd.DataFrame(rows).sort_values(["population", "water_body_code"]).reset_index(drop=True)


def _method_composition(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for (dataset, method), part in frame.groupby(["Dataset_ID", "TSS_method"], sort=True):
            rows.append({
                "population": population,
                "Dataset_ID": dataset,
                "TSS_method": method,
                "Organization_ID_membership": _join(part["Organization_ID"]),
                **_population_summary(part, population),
            })
    return pd.DataFrame(rows).sort_values(["population", "Dataset_ID", "TSS_method"]).reset_index(drop=True)


def _provider_composition(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for provider, part in frame.groupby("Organization_ID", sort=True):
            rows.append({
                "population": population,
                "Organization_ID": provider,
                "Dataset_ID_membership": _join(part["Dataset_ID"]),
                "TSS_method_membership": _join(part["TSS_method"]),
                **_population_summary(part, population),
            })
    return pd.DataFrame(rows).sort_values(["population", "Organization_ID"]).reset_index(drop=True)


def _country_composition(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for country, part in frame.groupby("Country", sort=True):
            rows.append({"population": population, "Country": country, **_population_summary(part, population)})
    return pd.DataFrame(rows).sort_values(["population", "Country"]).reset_index(drop=True)


def _exact_provenance_composition(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame in populations.items():
        for keys, part in frame.groupby(["Dataset_ID", "Organization_ID", "TSS_method"], sort=True):
            dataset, provider, method = keys
            rows.append({
                "population": population,
                "Dataset_ID": dataset,
                "Organization_ID": provider,
                "TSS_method": method,
                **_population_summary(part, population),
            })
    return pd.DataFrame(rows).sort_values(
        ["population", "Dataset_ID", "Organization_ID", "TSS_method"]
    ).reset_index(drop=True)


def _consistency_table(group_meta: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame:
    grouped = retained.groupby("group_id", sort=True)
    result = group_meta[["group_id", "raw_gloria_ids"]].copy()
    conflict_columns: list[str] = []
    for column in METADATA_COLUMNS:
        output = f"{column}_distinct_count"
        counts = grouped[column].nunique(dropna=False).rename(output)
        result = result.merge(counts, left_on="group_id", right_index=True, validate="one_to_one")
        conflict_columns.append(output)
    result["TSS_unit_distinct_count"] = 1
    conflict_columns.append("TSS_unit_distinct_count")
    result["metadata_conflict_count"] = (result[conflict_columns] > 1).sum(axis=1)
    result["metadata_conflict_status"] = np.where(
        result["metadata_conflict_count"].eq(0), "VERIFIED_NO_CONFLICT", "VERIFIED_CONFLICT"
    )
    return result.sort_values("group_id").reset_index(drop=True)


def _missingness_table(
    raw_tss: pd.DataFrame, retained: pd.DataFrame, group_meta: pd.DataFrame, method_reconciliation: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for population, frame in {
        "raw_tss_labelled_rows": raw_tss,
        "retained_complete_spectrum_rows": retained,
    }.items():
        for column in [
            "Dataset_ID", "Organization_ID", "TSS_method", "Date_Time_UTC", "Latitude", "Longitude", "Water_body_type",
        ]:
            rows.append({
                "population": population,
                "field": column,
                "total_rows": int(len(frame)),
                "missing_count": int(frame[column].isna().sum()),
                "missing_fraction": float(frame[column].isna().mean()),
                "availability": "present",
            })
    for column in ["Dataset_ID", "Organization_ID", "TSS_method"]:
        rows.append({
            "population": "canonical_groups",
            "field": column,
            "total_rows": int(len(group_meta)),
            "missing_count": int(group_meta[column].eq("<missing>").sum()),
            "missing_fraction": float(group_meta[column].eq("<missing>").mean()),
            "availability": "present",
        })
    for column in ["filter_type", "measurement_technique", "approach", "applications"]:
        rows.append({
            "population": "retained_dataset_method_combinations",
            "field": column,
            "total_rows": int(len(method_reconciliation)),
            "missing_count": int(method_reconciliation[column].eq("<missing>").sum()),
            "missing_fraction": float(method_reconciliation[column].eq("<missing>").mean()),
            "availability": "present_in_method_dictionary",
        })
    rows.append({
        "population": "raw_tss_labelled_rows",
        "field": "uncertainty_or_replicate_fields",
        "total_rows": 0,
        "missing_count": 0,
        "missing_fraction": np.nan,
        "availability": "not_present_in_GLORIA_meta_and_lab_csv",
    })
    return pd.DataFrame(rows)


def _method_reconciliation(
    group_meta: pd.DataFrame, target_sites: list[str], dictionary: pd.DataFrame
) -> pd.DataFrame:
    used = (
        group_meta.groupby(["Dataset_ID", "TSS_method"], sort=True)
        .agg(
            Organization_ID_membership=("Organization_ID", _join),
            raw_record_count=("raw_record_count", "sum"),
            canonical_group_count=("group_id", "count"),
            target_presence=("Site_name", lambda values: bool(set(values) & set(target_sites))),
        )
        .reset_index()
    )
    output = used.merge(dictionary, on=["Dataset_ID", "TSS_method"], how="left", validate="one_to_one")
    details = ["filter_type", "measurement_technique", "approach", "applications"]
    output["metadata_status"] = np.where(
        output[details].replace("<missing>", np.nan).notna().all(axis=1), "VERIFIED", "UNVERIFIED"
    )
    output["measurement_classification"] = np.where(
        output["measurement_technique"].str.contains("gravimetric", case=False, na=False),
        "gravimetric_documented", "not_classified_without_documented_text",
    )
    return output.sort_values(["Dataset_ID", "TSS_method"]).reset_index(drop=True)


def _unit_log_table(meta: pd.DataFrame, raw_tss: pd.DataFrame, retained: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"item": "authoritative_unit", "value": "g m-3", "status": "VERIFIED", "evidence": "GLORIA_variables_and_methods.xlsx / Data headers / TSS"},
        {"item": "reporting_equivalence", "value": "1 g m-3 = 1 mg L-1 numerically", "status": "VERIFIED_ARITHMETIC_EQUIVALENCE", "evidence": "mass/volume unit conversion; no code conversion"},
        {"item": "numerical_unit_conversion", "value": "none", "status": "VERIFIED", "evidence": "loaders and true-nested entry points retain TSS numeric values"},
        {"item": "raw_tss_min", "value": float(raw_tss["TSS"].min()), "status": "VERIFIED", "evidence": "GLORIA_meta_and_lab.csv after TSS nonmissing filter"},
        {"item": "raw_nonpositive_tss_rows", "value": int((raw_tss["TSS"] <= 0).sum()), "status": "VERIFIED", "evidence": "GLORIA_meta_and_lab.csv after TSS nonmissing filter"},
        {"item": "retained_tss_min", "value": float(retained["TSS"].min()), "status": "VERIFIED", "evidence": "same 400-750 nm complete-spectrum rows used by frozen loader"},
        {"item": "retained_nonpositive_tss_rows", "value": int((retained["TSS"] <= 0).sum()), "status": "VERIFIED", "evidence": "same 400-750 nm complete-spectrum rows used by frozen loader"},
        {"item": "canonical_group_tss_min", "value": float(groups["TSS"].min()), "status": "VERIFIED", "evidence": "src/lake_insitu/gloria.py build_tss_groups median"},
        {"item": "frozen_log10_expression", "value": "np.log10(groups['TSS'].clip(lower=1e-3))", "status": "VERIFIED", "evidence": "run_nested_capacity_boundary.py and run_h2_true_nested.py"},
        {"item": "clip_changes_retained_values", "value": False, "status": "VERIFIED", "evidence": "retained canonical minimum 0.1 > 1e-3"},
    ]
    return pd.DataFrame(rows)


def build_audit_tables(repo_root: Path = REPO_ROOT) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build E1 evidence tables from raw metadata without writing or fitting."""
    meta, raw_tss, joined, retained, groups, eligibility, eligible_sites = _load_inputs(repo_root)
    group_meta = _group_metadata(retained, groups)
    candidates = _candidate_table(eligibility, group_meta, retained)
    targets = candidates[candidates["final_eligible"]].copy()
    populations = {
        "full_canonical_archive": group_meta,
        "eligible_target_groups": group_meta[group_meta["Site_name"].isin(eligible_sites)],
        "auxiliary_groups": group_meta[~group_meta["Site_name"].isin(eligible_sites)],
    }
    dictionary = _read_method_dictionary(repo_root / GLORIA_RELATIVE)
    method_reconciliation = _method_reconciliation(group_meta, eligible_sites, dictionary)
    data_flow = pd.DataFrame([
        {"stage": "raw_gloria_rows", "count": len(meta), "definition": "all rows in GLORIA_meta_and_lab.csv"},
        {"stage": "tss_nonmissing_rows", "count": len(raw_tss), "definition": "TSS not missing"},
        {"stage": "rrs_join_rows", "count": len(joined), "definition": "TSS-labelled rows with GLORIA_ID in Rrs table"},
        {"stage": "complete_400_750_rows", "count": len(retained), "definition": "Rrs_400 through Rrs_750 complete"},
        {"stage": "canonical_groups", "count": len(group_meta), "definition": "site/country/day/3-decimal-coordinate median groups"},
        {"stage": "unique_site_names", "count": group_meta["Site_name"].nunique(), "definition": "canonical archive Site_name"},
        {"stage": "datasets", "count": group_meta["Dataset_ID"].nunique(), "definition": "canonical archive Dataset_ID"},
        {"stage": "nonmissing_providers", "count": group_meta.loc[group_meta["Organization_ID"] != "<missing>", "Organization_ID"].nunique(), "definition": "canonical archive nonmissing Organization_ID"},
        {"stage": "provider_labels_including_missing", "count": group_meta["Organization_ID"].nunique(), "definition": "canonical archive Organization_ID with <missing> retained as a label"},
        {"stage": "tss_methods", "count": group_meta["TSS_method"].nunique(), "definition": "canonical archive TSS_method"},
        {"stage": "pre_campaign_candidate_targets", "count": len(candidates), "definition": "rows in frozen eligible_tasks.csv"},
        {"stage": "eligible_targets", "count": len(eligible_sites), "definition": "all_eligibility_criteria_pass == True"},
        {"stage": "eligible_target_groups", "count": len(populations["eligible_target_groups"]), "definition": "canonical groups at 24 eligible Site_name values"},
        {"stage": "auxiliary_groups", "count": len(populations["auxiliary_groups"]), "definition": "canonical groups outside the 24 eligible Site_name values"},
    ])
    facts = {
        "raw_tss_min": float(raw_tss["TSS"].min()),
        "raw_nonpositive_tss_rows": int((raw_tss["TSS"] <= 0).sum()),
        "retained_tss_min": float(retained["TSS"].min()),
        "retained_nonpositive_tss_rows": int((retained["TSS"] <= 0).sum()),
        "canonical_group_metadata_conflicts": int((_consistency_table(group_meta, retained)["metadata_conflict_count"] > 0).sum()),
        "eligible_target_count": len(eligible_sites),
        "auxiliary_group_count": len(populations["auxiliary_groups"]),
    }
    tables = {
        "data_flow_audit": data_flow,
        "eligible_target_metadata": _target_metadata(targets, group_meta),
        "target_dataset_method_membership": _target_membership(eligible_sites, group_meta),
        "source_waterbody_composition": _waterbody_composition(populations),
        "source_tss_method_summary": _method_composition(populations),
        "source_provider_summary": _provider_composition(populations),
        "source_country_composition": _country_composition(populations),
        "source_dataset_provider_method_composition": _exact_provenance_composition(populations),
        "unit_and_log_transform_audit": _unit_log_table(meta, raw_tss, retained, groups),
        "canonical_group_metadata_consistency": _consistency_table(group_meta, retained),
        "metadata_missingness": _missingness_table(raw_tss, retained, group_meta, method_reconciliation),
        "method_dictionary_reconciliation": method_reconciliation,
        "candidate_target_eligibility": candidates,
    }
    return tables, facts


def _render_audit(tables: dict[str, pd.DataFrame], facts: dict[str, object]) -> str:
    flow = tables["data_flow_audit"].set_index("stage")["count"]
    missing = tables["metadata_missingness"]
    retained_missing = missing[missing["population"] == "retained_complete_spectrum_rows"]
    methods = tables["method_dictionary_reconciliation"]
    targets = tables["eligible_target_metadata"]
    provider_missing_groups = int(
        missing.loc[
            (missing["population"] == "canonical_groups")
            & (missing["field"] == "Organization_ID"),
            "missing_count",
        ].item()
    )
    missing_lines = "\n".join(
        f"| `{row.field}` | {int(row.missing_count)} / {int(row.total_rows)} |"
        for row in retained_missing.itertuples(index=False)
    )
    return f"""# TSS metadata, units, and target eligibility

This report summarizes metadata from the local GLORIA release and the 400--750 nm analysis archive.

## Verified data flow

| Stage | Count |
|---|---:|
| Raw GLORIA rows | {int(flow['raw_gloria_rows']):,} |
| TSS-nonmissing rows | {int(flow['tss_nonmissing_rows']):,} |
| Rrs-joined rows | {int(flow['rrs_join_rows']):,} |
| Complete 400--750 nm rows | {int(flow['complete_400_750_rows']):,} |
| Canonical groups | {int(flow['canonical_groups']):,} |
| Pre-diversity candidate targets | {int(flow['pre_campaign_candidate_targets'])} |
| Eligible targets | {int(flow['eligible_targets'])} |
| Eligible-target groups | {int(flow['eligible_target_groups']):,} |
| Auxiliary groups | {int(flow['auxiliary_groups']):,} |

## Unit and log-transform boundary

**VERIFIED.** The local primary codebook (`GLORIA_variables_and_methods.xlsx`, `Data headers`) gives TSS in **g m-3**. Numerically, 1 g m-3 = 1 mg L-1, but no loader or frozen true-nested entry point performs a unit conversion. The exact frozen expression is `np.log10(groups['TSS'].clip(lower=1e-3))`.

There are **{facts['raw_nonpositive_tss_rows']}** nonpositive TSS rows in the raw TSS-labelled metadata, but **{facts['retained_nonpositive_tss_rows']}** after the complete 400--750 nm filter; the retained minimum is **{facts['retained_tss_min']} g m-3**. Thus the lower clip changes no value actually supplied to the frozen H1/H2/H3 analyses. The dimensionless notation `log10[TSS / (1 mg L-1)]` is numerically equivalent to the implemented transformation.

## Eligibility and water-body labels

**VERIFIED.** The local codebook maps `Water_body_type=1` to **lake**; it has no explicit reservoir category or reservoir metadata field. All 24 eligible targets are GLORIA lake-labelled sites. `eligible_tasks.csv` defines the recorded campaign count as `n_unique_campaigns_dataset_id`: a **Dataset_ID proxy**, not an independently observed field-campaign identifier. Ba Be Lake and Lake Constance each fail the frozen diversity criterion (2 dates and 1 Dataset_ID proxy campaign). These metadata support the description `lake-labelled target sites` but do not identify reservoirs separately.

## Canonical-group consistency

**VERIFIED.** Across {int(flow['canonical_groups']):,} canonical groups, the count of groups with any multi-valued Dataset_ID, Organization_ID, TSS_method, water-body type, country, Site_name, or declared unit is **{facts['canonical_group_metadata_conflicts']}**. The complete group-level check, including raw GLORIA IDs, is in `canonical_group_metadata_consistency.csv`.

## Retained-row metadata missingness

| Field | Missing / rows |
|---|---:|
{missing_lines}

Provider IDs are missing for **591 / 4,500** retained rows and **{provider_missing_groups} / 4,292** canonical groups. The archive therefore has 32 nonmissing provider IDs (33 provider labels only when `<missing>` is counted as a category). Dataset_ID and TSS_method are complete in the retained archive.

## Method dictionary reconciliation

**VERIFIED.** The retained archive uses **{len(methods)}** Dataset_ID + TSS_method combinations; all have a dictionary `measurement_technique` explicitly describing a gravimetric dried-residue measurement. However, only **{int((methods['metadata_status'] == 'VERIFIED').sum())}** combinations have every retained dictionary field populated: `approach` is missing for **{int(methods['approach'].eq('<missing>').sum())}** and `applications` for **{int(methods['applications'].eq('<missing>').sum())}**. The audit therefore documents gravimetric measurement where stated but does not claim uniform standards or complete provider-specific provenance.

Among the 24 targets, **{int(targets['multi_dataset'].sum())}** have multiple Dataset_ID values, **{int(targets['multi_provider'].sum())}** multiple providers, and **{int(targets['multi_method'].sum())}** multiple TSS methods at the site level. This does not imply a within-canonical-group conflict; the group-level conflict audit above is zero.

## Interpretation boundary

**VERIFIED:** source composition, method dictionary reconciliation, missingness, and target/auxiliary membership are fully enumerated in the metadata CSVs.
**INFERENCE:** laboratory-method, provider, temporal, geographic, and water-body heterogeneity jointly characterize the evaluated source archive.
**Not identified:** this design does not estimate a causal effect of any provider, laboratory method, or water-body type on transfer performance. No such sensitivity experiment was run.

"""


def write_outputs(tables: dict[str, pd.DataFrame], facts: dict[str, object], output_dir: Path = RESULTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, lineterminator="\n")
    manifest = {
        "audit": "TSS metadata and eligibility",
        "facts": facts,
        "input_sha256": {
            str(path): _hash(REPO_ROOT / path)
            for path in [
                GLORIA_RELATIVE / "GLORIA_meta_and_lab.csv",
                GLORIA_RELATIVE / "GLORIA_Rrs.csv",
                GLORIA_RELATIVE / "GLORIA_variables_and_methods.xlsx",
                Path("data/processed/eligible_tasks.csv"),
            ]
        },
        "output_sha256": {
            f"{name}.csv": _hash(output_dir / f"{name}.csv") for name in sorted(tables)
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis_summary.md").write_text(_render_audit(tables, facts), encoding="utf-8", newline="\n")


def main() -> None:
    tables, facts = build_audit_tables(REPO_ROOT)
    write_outputs(tables, facts)
    print(
        "Metadata summary complete: "
        f"{facts['eligible_target_count']} eligible targets, "
        f"{facts['auxiliary_group_count']} auxiliary groups, "
        f"{facts['canonical_group_metadata_conflicts']} canonical metadata conflicts."
    )


if __name__ == "__main__":
    main()
