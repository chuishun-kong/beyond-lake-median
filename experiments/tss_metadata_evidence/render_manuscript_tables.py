"""Render deterministic supplementary metadata tables from metadata CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "tss_metadata_evidence" / "results"
OUTPUT = RESULTS / "manuscript_metadata_tables.md"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def _value(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _table(title: str, frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [f"**{title}**", "", "| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(_value(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def render() -> str:
    targets = _read("eligible_target_metadata.csv")
    targets = targets[[
        "Site_name", "Country", "water_body_labels", "raw_record_count", "canonical_group_count",
        "date_first", "date_last", "distinct_date_count", "campaign_count", "tss_min", "tss_median",
        "tss_max", "tss_unit", "Dataset_ID_membership", "Organization_ID_membership", "TSS_method_membership",
    ]].rename(columns={
        "water_body_labels": "Water-body label", "raw_record_count": "Raw records",
        "canonical_group_count": "Canonical groups", "date_first": "First date",
        "date_last": "Last date", "distinct_date_count": "Distinct dates",
        "campaign_count": "Distinct Dataset_ID", "tss_min": "TSS min", "tss_median": "TSS median",
        "tss_max": "TSS max", "tss_unit": "Unit", "Dataset_ID_membership": "Dataset membership",
        "Organization_ID_membership": "Provider membership", "TSS_method_membership": "TSS_method membership",
    })
    targets.insert(len(targets.columns), "Eligibility status", "included")
    flow = _read("data_flow_audit.csv")[["stage", "count"]].rename(columns={"stage": "Stage", "count": "Count"})
    flow["Stage"] = flow["Stage"].replace({"pre_campaign_candidate_targets": "pre-diversity candidate targets"})
    water = _read("source_waterbody_composition.csv")[["population", "water_body_label", "canonical_group_count", "site_count", "raw_record_count", "tss_min", "tss_median", "tss_max"]].rename(columns={
        "population": "Population", "water_body_label": "Water-body", "canonical_group_count": "Groups",
        "site_count": "Sites", "raw_record_count": "Raw records", "tss_min": "TSS min",
        "tss_median": "TSS median", "tss_max": "TSS max",
    })
    methods = _read("method_dictionary_reconciliation.csv")
    exact = _read("source_dataset_provider_method_composition.csv")
    membership = exact.pivot_table(index=["Dataset_ID", "TSS_method"], columns="population", values="canonical_group_count", aggfunc="sum", fill_value=0).reset_index()
    methods = methods.merge(membership, on=["Dataset_ID", "TSS_method"], how="left", validate="one_to_one")
    methods["Target/auxiliary"] = methods.apply(
        lambda row: "target + auxiliary" if row.get("eligible_target_groups", 0) > 0 and row.get("auxiliary_groups", 0) > 0 else ("target" if row.get("eligible_target_groups", 0) > 0 else "auxiliary"), axis=1
    )
    methods = methods[["Dataset_ID", "TSS_method", "Organization_ID_membership", "measurement_technique", "approach", "Target/auxiliary", "canonical_group_count", "metadata_status"]].rename(columns={
        "Organization_ID_membership": "Provider", "measurement_technique": "Documented technique",
        "approach": "Standard/reference", "canonical_group_count": "Groups",
        "metadata_status": "Metadata completeness",
    })
    missing = _read("metadata_missingness.csv")
    missing = missing[missing["field"].isin(["Organization_ID", "TSS_method", "Date_Time_UTC", "Latitude", "Longitude", "measurement_technique", "approach", "applications", "uncertainty_or_replicate_fields"])]
    missing = missing[["population", "field", "missing_count", "total_rows", "availability"]].rename(columns={
        "population": "Population", "field": "Field", "missing_count": "Missing", "total_rows": "Total", "availability": "Availability",
    })
    consistency = _read("canonical_group_metadata_consistency.csv")
    consistency_summary = pd.DataFrame([
        {"Check": "Canonical groups with any metadata conflict", "Count": int((consistency["metadata_conflict_count"] > 0).sum()), "Denominator": len(consistency)},
        {"Check": "Canonical groups with multi-Dataset_ID", "Count": int((consistency["Dataset_ID_distinct_count"] > 1).sum()), "Denominator": len(consistency)},
        {"Check": "Canonical groups with multi-provider", "Count": int((consistency["Organization_ID_distinct_count"] > 1).sum()), "Denominator": len(consistency)},
        {"Check": "Canonical groups with multi-TSS_method", "Count": int((consistency["TSS_method_distinct_count"] > 1).sum()), "Denominator": len(consistency)},
    ])
    sections = [
        "# Supplementary metadata tables\n\nGenerated deterministically from metadata CSVs; values are not hand-entered. The tables are descriptive and do not alter H1/H2/H3 results.",
        "## Supplementary Table S2 — Eligible target metadata\n\n" + _table("Table S2. Metadata for the 24 GLORIA lake-labelled target sites.", targets),
        "## Supplementary Table S3 — Data flow and source composition\n\n" + _table("Table S3A. Data-flow attrition and archive counts.", flow) + "\n\n" + _table("Table S3B. Water-body composition of the full, target, and auxiliary canonical archives.", water),
        "## Supplementary Table S4 — Dataset–method provenance\n\n" + _table("Table S4. Retained Dataset_ID × TSS_method combinations and dictionary evidence.", methods),
        "## Supplementary Table S5 — Missingness and group consistency\n\n" + _table("Table S5A. Metadata missingness and dictionary completeness.", missing) + "\n\n" + _table("Table S5B. Canonical-group consistency checks.", consistency_summary),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"Rendered {OUTPUT}")


if __name__ == "__main__":
    main()
