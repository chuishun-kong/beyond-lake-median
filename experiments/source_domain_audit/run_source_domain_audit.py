"""Deterministic, read-only audit of the frozen all-other-GLORIA source regime.

This script reconstructs the same 400--750 nm canonical TSS groups used by the
true-nested experiments.  It writes composition and exclusion manifests only;
it does not fit a model or calculate performance.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lake_insitu import gloria  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ELIGIBLE_PATH = REPO_ROOT / "data" / "processed" / "eligible_tasks.csv"
COORD_PRECISION = 3
WATER_BODY_TYPES = {
    1: "lake",
    2: "estuary",
    3: "coastal_ocean",
    4: "river",
    5: "other",
}


def _clean_value(value: object, missing: str = "<missing>") -> str:
    if pd.isna(value):
        return missing
    text = str(value).strip()
    return text if text else missing


def _sorted_join(values: Iterable[object]) -> str:
    return "|".join(sorted({_clean_value(value) for value in values}))


def _split_unique(values: Iterable[object]) -> list[str]:
    items: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        items.update(part for part in str(value).split("|") if part)
    return sorted(items)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().eq("true")


def normalize_site_name(name: str) -> str:
    """Conservative name normalization for collision discovery, not identity."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    generic = {"lake", "reservoir", "the"}
    retained = [token for token in tokens if token not in generic]
    return " ".join(retained)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return math.nan
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def classify_alias_candidate(
    *,
    target_name: str,
    target_country: str,
    target_lat: float,
    target_lon: float,
    candidate_name: str,
    candidate_country: str,
    candidate_lat: float,
    candidate_lon: float,
    reasons: tuple[str, ...],
) -> dict[str, str]:
    """Classify identity evidence without treating name similarity as proof."""
    if target_name == candidate_name:
        return {
            "identity_class": "confirmed_target_identity",
            "evidence_status": "VERIFIED",
            "resolution": "excluded_by_exact_Site_name",
        }

    distance_km = _haversine_km(target_lat, target_lon, candidate_lat, candidate_lon)
    same_country = target_country == candidate_country
    normalized_equal = normalize_site_name(target_name) == normalize_site_name(candidate_name)

    if normalized_equal and not same_country and not pd.isna(distance_km) and distance_km > 100:
        return {
            "identity_class": "non_match",
            "evidence_status": "VERIFIED",
            "resolution": "same_normalized_name_but_different_country_and_distant_coordinates",
        }

    if "same_rounded_coordinate" in reasons or (normalized_equal and same_country):
        return {
            "identity_class": "candidate_alias",
            "evidence_status": "UNVERIFIED",
            "resolution": "manual_identity_review_required",
        }

    return {
        "identity_class": "candidate_alias",
        "evidence_status": "UNVERIFIED",
        "resolution": "name_similarity_is_discovery_evidence_only",
    }


def _canonical_group_key(meta: pd.DataFrame) -> pd.Series:
    date_day = meta["Date_Time_UTC"].dt.floor("D")
    return (
        meta["Site_name"].astype(str)
        + "||"
        + meta["Country"].astype(str)
        + "||"
        + date_day.astype(str)
        + "||"
        + meta["Latitude"].round(COORD_PRECISION).astype(str)
        + "||"
        + meta["Longitude"].round(COORD_PRECISION).astype(str)
    )


def load_audit_inputs(repo_root: Path = REPO_ROOT):
    """Load canonical groups and attach raw provenance to each retained group."""
    gloria_dir = repo_root / "data" / "raw" / "gloria" / "GLORIA_2022"
    meta = gloria.load_meta(gloria_dir)
    rrs = gloria.load_rrs(gloria_dir)
    band_cols = gloria.rrs_band_columns(rrs, lo=400, hi=750)
    groups = gloria.build_tss_groups(meta, rrs, band_cols, coord_precision=COORD_PRECISION)

    tss = meta[meta["TSS"].notna()].copy()
    tss["date_day"] = tss["Date_Time_UTC"].dt.floor("D")
    tss["group_id"] = _canonical_group_key(tss)
    raw = tss.merge(rrs[["GLORIA_ID"]], on="GLORIA_ID", how="inner")
    raw = raw[raw["group_id"].isin(set(groups["group_id"]))].copy()
    raw["latitude_rounded"] = raw["Latitude"].round(COORD_PRECISION)
    raw["longitude_rounded"] = raw["Longitude"].round(COORD_PRECISION)

    provenance = (
        raw.groupby("group_id", sort=True)
        .agg(
            Site_name=("Site_name", "first"),
            Country=("Country", "first"),
            Water_body_type=("Water_body_type", lambda s: int(s.mode().iloc[0])),
            date_day=("date_day", "first"),
            latitude_rounded=("latitude_rounded", "first"),
            longitude_rounded=("longitude_rounded", "first"),
            dataset_ids=("Dataset_ID", _sorted_join),
            organization_ids=("Organization_ID", _sorted_join),
            tss_methods=("TSS_method", _sorted_join),
            raw_record_count=("GLORIA_ID", "count"),
        )
        .reset_index()
    )

    if set(provenance["group_id"]) != set(groups["group_id"]):
        raise AssertionError("Raw provenance does not reconcile with canonical retained groups")

    eligible = pd.read_csv(repo_root / "data" / "processed" / "eligible_tasks.csv")
    eligible_sites = eligible.loc[_as_bool(eligible["all_eligibility_criteria_pass"]), "lake"].tolist()
    return groups, provenance, eligible_sites


def _site_summary(provenance: pd.DataFrame) -> pd.DataFrame:
    return (
        provenance.groupby("Site_name", sort=True)
        .agg(
            countries=("Country", _sorted_join),
            water_body_types=("Water_body_type", lambda s: "|".join(map(str, sorted(set(s))))),
            centroid_lat=("latitude_rounded", "median"),
            centroid_lon=("longitude_rounded", "median"),
            group_count=("group_id", "nunique"),
            dataset_ids=("dataset_ids", lambda s: "|".join(_split_unique(s))),
            organization_ids=("organization_ids", lambda s: "|".join(_split_unique(s))),
            tss_methods=("tss_methods", lambda s: "|".join(_split_unique(s))),
        )
        .reset_index()
    )


def _alias_table(provenance: pd.DataFrame, eligible_sites: list[str]) -> pd.DataFrame:
    sites = _site_summary(provenance).set_index("Site_name")
    coordinates = {
        name: set(zip(frame["latitude_rounded"], frame["longitude_rounded"]))
        for name, frame in provenance.groupby("Site_name", sort=True)
    }
    rows: list[dict[str, object]] = []

    for target in sorted(eligible_sites):
        target_row = sites.loc[target]
        target_country = target_row["countries"]
        exact = classify_alias_candidate(
            target_name=target,
            target_country=target_country,
            target_lat=target_row["centroid_lat"],
            target_lon=target_row["centroid_lon"],
            candidate_name=target,
            candidate_country=target_country,
            candidate_lat=target_row["centroid_lat"],
            candidate_lon=target_row["centroid_lon"],
            reasons=("exact_site_name",),
        )
        rows.append(
            {
                "outer_lake": target,
                "normalized_outer_name": normalize_site_name(target),
                "candidate_site": target,
                "normalized_candidate_name": normalize_site_name(target),
                "candidate_in_outer_source": False,
                "discovery_reasons": "exact_site_name",
                "name_similarity": 1.0,
                "same_country": True,
                "centroid_distance_km": 0.0,
                "shared_rounded_coordinates": len(coordinates[target]),
                "target_dataset_ids": target_row["dataset_ids"],
                "candidate_dataset_ids": target_row["dataset_ids"],
                **exact,
            }
        )

        for candidate, candidate_row in sites.iterrows():
            if candidate == target:
                continue
            reasons: list[str] = []
            if normalize_site_name(target) == normalize_site_name(candidate):
                reasons.append("normalized_name_collision")
            shared_coordinates = coordinates[target] & coordinates[candidate]
            if shared_coordinates:
                reasons.append("same_rounded_coordinate")
            similarity = SequenceMatcher(None, normalize_site_name(target), normalize_site_name(candidate)).ratio()
            distance_km = _haversine_km(
                target_row["centroid_lat"],
                target_row["centroid_lon"],
                candidate_row["centroid_lat"],
                candidate_row["centroid_lon"],
            )
            if (
                similarity >= 0.90
                and target_row["countries"] == candidate_row["countries"]
                and not pd.isna(distance_km)
                and distance_km <= 25
            ):
                reasons.append("high_name_similarity_same_country_nearby")
            if not reasons:
                continue

            finding = classify_alias_candidate(
                target_name=target,
                target_country=target_row["countries"],
                target_lat=target_row["centroid_lat"],
                target_lon=target_row["centroid_lon"],
                candidate_name=candidate,
                candidate_country=candidate_row["countries"],
                candidate_lat=candidate_row["centroid_lat"],
                candidate_lon=candidate_row["centroid_lon"],
                reasons=tuple(sorted(set(reasons))),
            )
            rows.append(
                {
                    "outer_lake": target,
                    "normalized_outer_name": normalize_site_name(target),
                    "candidate_site": candidate,
                    "normalized_candidate_name": normalize_site_name(candidate),
                    "candidate_in_outer_source": True,
                    "discovery_reasons": "|".join(sorted(set(reasons))),
                    "name_similarity": similarity,
                    "same_country": target_row["countries"] == candidate_row["countries"],
                    "centroid_distance_km": distance_km,
                    "shared_rounded_coordinates": len(shared_coordinates),
                    "target_dataset_ids": target_row["dataset_ids"],
                    "candidate_dataset_ids": candidate_row["dataset_ids"],
                    **finding,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["outer_lake", "candidate_in_outer_source", "identity_class", "candidate_site"]
    ).reset_index(drop=True)


def build_audit_tables(
    groups: pd.DataFrame, provenance: pd.DataFrame, eligible_sites: list[str]
) -> dict[str, pd.DataFrame]:
    """Build all Gate B manifests from canonical groups and raw provenance."""
    eligible_set = set(eligible_sites)
    canonical_group_count = len(provenance)
    canonical_site_count = provenance["Site_name"].nunique()
    eligible_archive = provenance[provenance["Site_name"].isin(eligible_set)]
    auxiliary_archive = provenance[~provenance["Site_name"].isin(eligible_set)]
    alias = _alias_table(provenance, eligible_sites)
    cross_dataset_collisions = provenance["dataset_ids"].str.contains(r"\|", regex=True).sum()

    composition_rows: list[dict[str, object]] = []
    source_site_frames: list[pd.DataFrame] = []
    tss_frames: list[pd.DataFrame] = []
    inner_rows: list[dict[str, object]] = []

    for outer in sorted(eligible_sites):
        target = provenance[provenance["Site_name"] == outer]
        source = provenance[provenance["Site_name"] != outer].copy()
        source_auxiliary = source[~source["Site_name"].isin(eligible_set)]
        source_eligible = source[source["Site_name"].isin(eligible_set)]
        outer_alias = alias[(alias["outer_lake"] == outer) & alias["candidate_in_outer_source"]]

        row: dict[str, object] = {
            "outer_lake": outer,
            "normalized_outer_name": normalize_site_name(outer),
            "outer_country": _sorted_join(target["Country"]),
            "target_group_count": target["group_id"].nunique(),
            "canonical_group_count": canonical_group_count,
            "canonical_site_count": canonical_site_count,
            "eligible_lake_count": len(eligible_sites),
            "eligible_archive_group_count": len(eligible_archive),
            "auxiliary_archive_group_count": len(auxiliary_archive),
            "auxiliary_archive_site_count": auxiliary_archive["Site_name"].nunique(),
            "source_group_count": len(source),
            "source_site_count": source["Site_name"].nunique(),
            "source_dataset_count": len(_split_unique(source["dataset_ids"])),
            "source_provider_count": len(_split_unique(source["organization_ids"])),
            "source_tss_method_count": len(_split_unique(source["tss_methods"])),
            "eligible_source_group_count": len(source_eligible),
            "eligible_source_site_count": source_eligible["Site_name"].nunique(),
            "auxiliary_source_group_count": len(source_auxiliary),
            "auxiliary_source_site_count": source_auxiliary["Site_name"].nunique(),
            "target_groups_in_source": source.loc[source["Site_name"] == outer, "group_id"].nunique(),
            "inner_validation_unit_count": len(eligible_sites) - 1,
            "confirmed_alias_leakage_count": int(
                ((outer_alias["identity_class"] == "confirmed_target_identity")).sum()
            ),
            "unresolved_alias_candidate_count": int(
                ((outer_alias["identity_class"] == "candidate_alias")).sum()
            ),
            "verified_non_match_count": int((outer_alias["identity_class"] == "non_match").sum()),
            "archive_cross_dataset_group_key_collision_count": int(cross_dataset_collisions),
        }
        for code, label in WATER_BODY_TYPES.items():
            typed = source[source["Water_body_type"] == code]
            row[f"source_type_{code}_{label}_group_count"] = len(typed)
            row[f"source_type_{code}_{label}_site_count"] = typed["Site_name"].nunique()
        composition_rows.append(row)

        site_table = (
            source.groupby("Site_name", sort=True)
            .agg(
                countries=("Country", _sorted_join),
                water_body_type_codes=(
                    "Water_body_type",
                    lambda s: "|".join(map(str, sorted(set(s)))),
                ),
                group_count=("group_id", "nunique"),
                dataset_ids=("dataset_ids", lambda s: "|".join(_split_unique(s))),
                organization_ids=("organization_ids", lambda s: "|".join(_split_unique(s))),
                tss_methods=("tss_methods", lambda s: "|".join(_split_unique(s))),
                raw_record_count=("raw_record_count", "sum"),
                centroid_lat=("latitude_rounded", "median"),
                centroid_lon=("longitude_rounded", "median"),
            )
            .reset_index()
            .rename(columns={"Site_name": "source_site"})
        )
        site_table.insert(0, "outer_lake", outer)
        site_table.insert(2, "normalized_source_site", site_table["source_site"].map(normalize_site_name))
        site_table["eligible_target_site"] = site_table["source_site"].isin(eligible_set)
        site_table["auxiliary_source_site"] = ~site_table["eligible_target_site"]
        site_table["exact_name_collision_with_outer"] = site_table["source_site"].eq(outer)
        site_table["normalized_name_collision_with_outer"] = site_table[
            "normalized_source_site"
        ].eq(normalize_site_name(outer))
        source_site_frames.append(site_table)

        tss_table = (
            source.groupby(
                ["Water_body_type", "dataset_ids", "organization_ids", "tss_methods"],
                sort=True,
                dropna=False,
            )
            .agg(
                group_count=("group_id", "nunique"),
                site_count=("Site_name", "nunique"),
                raw_record_count=("raw_record_count", "sum"),
                auxiliary_group_count=("Site_name", lambda s: (~s.isin(eligible_set)).sum()),
            )
            .reset_index()
        )
        tss_table.insert(0, "outer_lake", outer)
        tss_table.insert(
            2,
            "water_body_type_label",
            tss_table["Water_body_type"].map(WATER_BODY_TYPES),
        )
        tss_frames.append(tss_table)

        for inner in sorted(eligible_set - {outer}):
            training = provenance[~provenance["Site_name"].isin({outer, inner})]
            inner_rows.append(
                {
                    "outer_lake": outer,
                    "inner_validation_lake": inner,
                    "inner_training_group_count": len(training),
                    "inner_training_site_count": training["Site_name"].nunique(),
                    "outer_groups_in_inner_training": training.loc[
                        training["Site_name"] == outer, "group_id"
                    ].nunique(),
                    "inner_validation_groups_in_inner_training": training.loc[
                        training["Site_name"] == inner, "group_id"
                    ].nunique(),
                    "inner_validation_group_count": provenance.loc[
                        provenance["Site_name"] == inner, "group_id"
                    ].nunique(),
                    "exclusion_status": "VERIFIED",
                }
            )

    tables = {
        "source_composition_by_outer_lake": pd.DataFrame(composition_rows).sort_values(
            "outer_lake"
        ).reset_index(drop=True),
        "source_sites_by_outer_lake": pd.concat(source_site_frames, ignore_index=True).sort_values(
            ["outer_lake", "source_site"]
        ).reset_index(drop=True),
        "target_alias_audit": alias,
        "tss_method_composition": pd.concat(tss_frames, ignore_index=True).sort_values(
            ["outer_lake", "Water_body_type", "dataset_ids", "organization_ids", "tss_methods"]
        ).reset_index(drop=True),
        "inner_fold_exclusion_audit": pd.DataFrame(inner_rows).sort_values(
            ["outer_lake", "inner_validation_lake"]
        ).reset_index(drop=True),
    }

    composition = tables["source_composition_by_outer_lake"]
    inner = tables["inner_fold_exclusion_audit"]
    if len(composition) != 24 or len(inner) != 24 * 23:
        raise AssertionError("Expected 24 outer rows and 552 inner exclusion rows")
    if composition["target_groups_in_source"].any():
        raise AssertionError("At least one outer target remains in its source fit")
    if inner[
        ["outer_groups_in_inner_training", "inner_validation_groups_in_inner_training"]
    ].to_numpy().any():
        raise AssertionError("At least one inner training fit violates lake exclusion")
    return tables


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path = RESULTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(
            output_dir / f"{name}.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            float_format="%.6f",
        )


def main() -> None:
    groups, provenance, eligible_sites = load_audit_inputs(REPO_ROOT)
    tables = build_audit_tables(groups, provenance, eligible_sites)
    write_tables(tables)
    composition = tables["source_composition_by_outer_lake"]
    print(
        "Gate B source-domain audit complete: "
        f"{len(groups)} canonical groups, {len(eligible_sites)} outer lakes, "
        f"{composition['auxiliary_archive_group_count'].iloc[0]} auxiliary groups, "
        f"{len(tables['inner_fold_exclusion_audit'])} inner exclusions checked."
    )


if __name__ == "__main__":
    main()
