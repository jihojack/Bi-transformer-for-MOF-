#!/usr/bin/env python3
"""Build a paired 50K CO2/CH4 adsorption dataset from MOFX-DB.

All adsorption values are expressed in mmol/g. Source values stored as mol/kg
are numerically identical because 1 mol/kg == 1 mmol/g.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_ZIP_NAME = "bulk-dl-mofdb-version-dc8a0295db.zip"
EXISTING_CSV_NAME = "MOF_CO2_30K_light.csv"
OUTPUT_CSV_NAME = "MOF_CO2_CH4_50K.csv"
OUTPUT_CIF_ZIP_NAME = "MOF_CO2_CH4_50K_CIF.zip"
OUTPUT_SPLIT_NAME = "MOF_CO2_CH4_50K_split_70_10_20.json"
OUTPUT_MANIFEST_NAME = "MOF_CO2_CH4_50K_manifest.json"

TARGET_SIZE = 50_000
RANDOM_SEED = 42
TEMPERATURE_K = 298.0
MAX_CO2_Q_0P5_MMOL_G = 20.0

GAS_SPECS = {
    "CO2": {
        "pressures": (0.01, 0.05, 0.1, 0.5, 2.5),
        "columns": (
            "co2_q_0p01bar",
            "co2_q_0p05bar",
            "co2_adsorption_0.1bar",
            "co2_adsorption_0.5bar",
            "co2_adsorption_2.5bar",
        ),
    },
    "CH4": {
        "pressures": (0.05, 0.5, 0.9, 2.5, 4.5),
        "columns": (
            "ch4_adsorption_0.05bar",
            "ch4_adsorption_0.5bar",
            "ch4_adsorption_0.9bar",
            "ch4_adsorption_2.5bar",
            "ch4_adsorption_4.5bar",
        ),
    },
}

CO2_COLUMNS = GAS_SPECS["CO2"]["columns"]
CH4_COLUMNS = GAS_SPECS["CH4"]["columns"]
TARGET_COLUMNS = (*CO2_COLUMNS, *CH4_COLUMNS)
OUTPUT_COLUMNS = (
    "mof_id",
    "surface_area_m2g",
    "void_fraction",
    "lcd",
    "pld",
    "elements",
    "has_cif",
    *TARGET_COLUMNS,
)

ARCHIVE_MEMBER_RE = re.compile(
    r"^hmof-(?P<number>\d+)-\(id:(?P<internal_id>\d+)\)"
    r"\.(?P<extension>json|cif)$",
    flags=re.IGNORECASE,
)
MOF_ID_RE = re.compile(r"^hmof-(\d+)$", flags=re.IGNORECASE)


def discover_file(name: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd() / name, root / "data" / "raw" / name,
                  root / "data" / "processed" / name]
    matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name}; found {matches}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=None)
    parser.add_argument("--existing-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "MOF_50K_build")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def canonical_mof_id(value: object) -> str | None:
    match = MOF_ID_RE.fullmatch(str(value).strip())
    return f"hmof-{int(match.group(1))}" if match else None


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def pressure_to_bar(value: object, unit: object) -> float | None:
    pressure = finite_float(value)
    if pressure is None:
        return None
    normalized = str(unit).strip().lower().replace(" ", "")
    if normalized in {"bar", "bars"}:
        return pressure
    if normalized in {"pa", "pascal", "pascals"}:
        return pressure / 100_000.0
    if normalized in {"kpa", "kilopascal", "kilopascals"}:
        return pressure / 100.0
    return None


def gas_formula(isotherm: dict) -> str | None:
    formulas = []
    for adsorbate in isotherm.get("adsorbates") or []:
        if isinstance(adsorbate, dict):
            formulas.append(str(adsorbate.get("formula", "")).strip().upper())
        else:
            formulas.append(str(adsorbate).strip().upper())
    return formulas[0] if len(formulas) == 1 else None


def adsorption_unit_is_compatible(unit: object) -> bool:
    normalized = (
        str(unit).strip().lower().replace(" ", "").replace("^-1", "-1")
    )
    return normalized in {"mol/kg", "molkg-1", "mmol/g", "mmolg-1"}


def extract_curve(payload: dict, gas: str) -> tuple[float, ...] | None:
    spec = GAS_SPECS[gas]
    complete_curves = []
    for isotherm in payload.get("isotherms") or []:
        if not isinstance(isotherm, dict) or gas_formula(isotherm) != gas:
            continue
        temperature = finite_float(isotherm.get("temperature"))
        if temperature is None or not math.isclose(
            temperature, TEMPERATURE_K, abs_tol=0.5
        ):
            continue
        if not adsorption_unit_is_compatible(isotherm.get("adsorptionUnits")):
            continue

        matched: dict[float, tuple[float, float]] = {}
        for point in isotherm.get("isotherm_data") or []:
            if not isinstance(point, dict):
                continue
            pressure = pressure_to_bar(
                point.get("pressure"), isotherm.get("pressureUnits")
            )
            uptake = finite_float(point.get("total_adsorption"))
            if pressure is None or uptake is None:
                continue
            for target_pressure in spec["pressures"]:
                distance = abs(pressure - target_pressure)
                tolerance = max(1e-7, target_pressure * 1e-4)
                if distance <= tolerance:
                    previous = matched.get(target_pressure)
                    if previous is None or distance < previous[0]:
                        matched[target_pressure] = (distance, uptake)

        if len(matched) == len(spec["pressures"]):
            complete_curves.append(
                tuple(matched[p][1] for p in spec["pressures"])
            )
    return complete_curves[0] if complete_curves else None


def extract_elements(payload: dict) -> str:
    symbols = []
    for element in payload.get("elements") or []:
        symbol = (
            element.get("symbol") or element.get("name")
            if isinstance(element, dict)
            else element
        )
        text = str(symbol).strip() if symbol is not None else ""
        if text and text not in symbols:
            symbols.append(text)
    return ",".join(symbols) if symbols else "Unknown"


def record_from_json(payload: dict, mof_id: str, cif_member: str) -> tuple[dict | None, str]:
    descriptors = {
        "surface_area_m2g": finite_float(payload.get("surface_area_m2g")),
        "void_fraction": finite_float(payload.get("void_fraction")),
        "lcd": finite_float(payload.get("lcd")),
        "pld": finite_float(payload.get("pld")),
    }
    if any(value is None for value in descriptors.values()):
        return None, "missing_descriptor"
    if (
        descriptors["surface_area_m2g"] <= 0
        or descriptors["void_fraction"] < 0
        or descriptors["lcd"] <= 0
        or descriptors["pld"] <= 0
    ):
        return None, "invalid_descriptor"

    co2 = extract_curve(payload, "CO2")
    ch4 = extract_curve(payload, "CH4")
    if co2 is None:
        return None, "missing_co2_curve"
    if ch4 is None:
        return None, "missing_ch4_curve"
    if any(value < 0 for value in (*co2, *ch4)):
        return None, "negative_uptake"
    if any(co2[i] > co2[i + 1] + 1e-8 for i in range(len(co2) - 1)):
        return None, "nonmonotonic_co2"
    if any(ch4[i] > ch4[i + 1] + 1e-8 for i in range(len(ch4) - 1)):
        return None, "nonmonotonic_ch4"
    if co2[3] > MAX_CO2_Q_0P5_MMOL_G:
        return None, "co2_0p5bar_outlier"

    record = {
        "mof_id": mof_id,
        **descriptors,
        "elements": extract_elements(payload),
        "has_cif": True,
        **dict(zip(CO2_COLUMNS, co2)),
        **dict(zip(CH4_COLUMNS, ch4)),
        "_cif_member": cif_member,
    }
    return record, "valid"


def index_archive(archive: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, str]]:
    json_members = {}
    cif_members = {}
    for info in archive.infolist():
        match = ARCHIVE_MEMBER_RE.fullmatch(Path(info.filename).name)
        if not match:
            continue
        mof_id = f"hmof-{int(match.group('number'))}"
        destination = (
            json_members
            if match.group("extension").lower() == "json"
            else cif_members
        )
        if mof_id in destination:
            raise RuntimeError(f"Duplicate archive member for {mof_id}")
        destination[mof_id] = info.filename
    return json_members, cif_members


def scan_valid_records(archive, json_members, cif_members):
    valid_records = []
    counts = Counter()
    for index, (mof_id, json_member) in enumerate(json_members.items(), start=1):
        cif_member = cif_members.get(mof_id)
        if cif_member is None:
            counts["missing_cif"] += 1
            continue
        try:
            payload = json.loads(archive.read(json_member))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            counts["invalid_json"] += 1
            continue
        record, status = record_from_json(payload, mof_id, cif_member)
        counts[status] += 1
        if record is not None:
            valid_records.append(record)
        if index % 10_000 == 0:
            print(
                f"Scanned {index:,}/{len(json_members):,}; "
                f"paired valid={counts['valid']:,}",
                flush=True,
            )
    return valid_records, counts


def load_existing(existing_csv: Path, valid_by_id: dict[str, dict]):
    existing = pd.read_csv(existing_csv)
    existing_ids = [canonical_mof_id(value) for value in existing["mof_id"]]
    if any(value is None for value in existing_ids):
        raise ValueError("Existing CSV contains an invalid mof_id")
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("Existing CSV contains duplicate mof_id values")

    retained = [mof_id for mof_id in existing_ids if mof_id in valid_by_id]
    rejected = [mof_id for mof_id in existing_ids if mof_id not in valid_by_id]
    old_by_id = existing.assign(_canonical_id=existing_ids).set_index("_canonical_id")
    max_differences = {}
    for column in CO2_COLUMNS:
        regenerated = pd.Series(
            {mof_id: valid_by_id[mof_id][column] for mof_id in retained}
        )
        original = pd.to_numeric(old_by_id.loc[regenerated.index, column], errors="coerce")
        difference = (original - regenerated).abs().max()
        max_differences[column] = float(difference)
        if not math.isfinite(difference) or difference > 1e-6:
            raise RuntimeError(f"Existing target mismatch in {column}: {difference}")
    return existing, existing_ids, retained, rejected, max_differences


def quantile_strata(frame: pd.DataFrame) -> pd.Series:
    columns = (
        CO2_COLUMNS[0],
        CO2_COLUMNS[-1],
        CH4_COLUMNS[0],
        CH4_COLUMNS[-1],
    )
    labels = []
    for column in columns:
        labels.append(
            pd.qcut(
                frame[column].rank(method="first"),
                q=5,
                labels=False,
                duplicates="drop",
            ).astype(str)
        )
    return labels[0] + "_" + labels[1] + "_" + labels[2] + "_" + labels[3]


def allocate_sqrt_quotas(counts: pd.Series, sample_size: int) -> pd.Series:
    weights = np.sqrt(counts.astype(float))
    raw = weights / weights.sum() * sample_size
    quotas = np.minimum(np.floor(raw).astype(int), counts.astype(int))
    remaining = int(sample_size - quotas.sum())
    while remaining > 0:
        available = counts > quotas
        if not available.any():
            raise RuntimeError("Not enough candidates for requested sample")
        score = (raw - quotas).where(available, -np.inf)
        quotas.loc[score.idxmax()] += 1
        remaining -= 1
    return quotas


def select_records(valid_records, retained_ids, target_size, seed):
    valid_by_id = {record["mof_id"]: record for record in valid_records}
    retained_set = set(retained_ids)
    add_count = target_size - len(retained_ids)
    candidates = pd.DataFrame(
        [record for record in valid_records if record["mof_id"] not in retained_set]
    )
    if len(candidates) < add_count:
        raise RuntimeError(
            f"Need {add_count:,} additional rows; only {len(candidates):,} available"
        )
    candidates["_stratum"] = quantile_strata(candidates)
    counts = candidates["_stratum"].value_counts().sort_index()
    quotas = allocate_sqrt_quotas(counts, add_count)
    sampled = []
    for group_number, (stratum, quota) in enumerate(quotas.items()):
        group = candidates[candidates["_stratum"] == stratum]
        sampled.append(
            group.sample(n=int(quota), random_state=seed + group_number)
        )
    added = pd.concat(sampled, ignore_index=True)
    added = added.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected_ids = retained_ids + added["mof_id"].tolist()
    selected = [valid_by_id[mof_id] for mof_id in selected_ids]
    return selected, added["mof_id"].tolist(), {
        "paired_valid_candidates": len(valid_records),
        "existing_rows_retained": len(retained_ids),
        "new_rows_selected": add_count,
        "remaining_candidates_before_sampling": len(candidates),
        "occupied_strata": int(len(counts)),
        "sampling": (
            "Square-root allocation over 5-bin quantile strata for CO2 at "
            "0.01/2.5 bar and CH4 at 0.05/4.5 bar; seed fixed."
        ),
    }


def reconstruct_old_split(existing: pd.DataFrame, seed: int) -> dict[str, str]:
    indices = np.arange(len(existing), dtype=np.int64)
    bins = pd.qcut(
        existing[CO2_COLUMNS[3]].rank(method="first"),
        q=20,
        labels=False,
    ).to_numpy()
    trainval, test = train_test_split(
        indices, test_size=0.20, random_state=seed, shuffle=True, stratify=bins
    )
    train, val = train_test_split(
        trainval,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=bins[trainval],
    )
    mapping = {}
    ids = [canonical_mof_id(value) for value in existing["mof_id"]]
    for split, split_indices in (("train", train), ("val", val), ("test", test)):
        for index in split_indices:
            mapping[ids[index]] = split
    return mapping


def make_fixed_split(selected: list[dict], retained_ids, old_split, seed):
    target_counts = {
        "train": int(len(selected) * 0.70),
        "val": int(len(selected) * 0.10),
    }
    target_counts["test"] = len(selected) - target_counts["train"] - target_counts["val"]
    assignment = {mof_id: old_split[mof_id] for mof_id in retained_ids}
    retained_counts = Counter(assignment.values())
    needed = {
        split: target_counts[split] - retained_counts[split]
        for split in target_counts
    }
    if any(value < 0 for value in needed.values()):
        raise RuntimeError(f"Retained split exceeds final target counts: {needed}")

    selected_frame = pd.DataFrame(selected)
    new_frame = selected_frame[~selected_frame["mof_id"].isin(retained_ids)].copy()
    combined_rank = (
        new_frame[CO2_COLUMNS[3]].rank(method="first", pct=True)
        + new_frame[CH4_COLUMNS[1]].rank(method="first", pct=True)
    ) / 2.0
    strata = pd.qcut(combined_rank.rank(method="first"), q=20, labels=False)
    new_indices = np.arange(len(new_frame), dtype=np.int64)
    trainval, test = train_test_split(
        new_indices,
        test_size=needed["test"],
        random_state=seed,
        shuffle=True,
        stratify=strata.to_numpy(),
    )
    remaining_strata = strata.to_numpy()[trainval]
    train, val = train_test_split(
        trainval,
        test_size=needed["val"],
        random_state=seed,
        shuffle=True,
        stratify=remaining_strata,
    )
    new_ids = new_frame["mof_id"].tolist()
    for split, split_indices in (("train", train), ("val", val), ("test", test)):
        for index in split_indices:
            assignment[new_ids[index]] = split

    selected_ids = selected_frame["mof_id"].tolist()
    payload = {
        "seed": seed,
        "strategy": (
            "Retained 30K IDs keep the exact original seed-42 stratified split; "
            "new IDs fill each split to 70/10/20 using joint CO2/CH4 0.5-bar strata."
        ),
    }
    for split in ("train", "val", "test"):
        indices = [i for i, mof_id in enumerate(selected_ids) if assignment[mof_id] == split]
        payload[f"{split}_indices"] = indices
        payload[f"{split}_mof_ids"] = [selected_ids[i] for i in indices]
    payload["counts"] = {
        split: len(payload[f"{split}_indices"])
        for split in ("train", "val", "test")
    }
    if payload["counts"] != target_counts:
        raise RuntimeError(f"Wrong split counts: {payload['counts']} != {target_counts}")
    return payload


def write_outputs(archive, selected, split_payload, output_dir, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_CSV_NAME
    cif_zip_path = output_dir / OUTPUT_CIF_ZIP_NAME
    split_path = output_dir / OUTPUT_SPLIT_NAME
    manifest_path = output_dir / OUTPUT_MANIFEST_NAME

    frame = pd.DataFrame(selected)[list(OUTPUT_COLUMNS)]
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    with zipfile.ZipFile(
        cif_zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as destination:
        for index, record in enumerate(selected, start=1):
            member = record["_cif_member"]
            destination.writestr(Path(member).name, archive.read(member))
            if index % 5_000 == 0:
                print(f"Wrote {index:,}/{len(selected):,} CIF files", flush=True)

    manifest["outputs"] = {
        "csv": csv_path.name,
        "cif_zip": cif_zip_path.name,
        "split": split_path.name,
        "manifest": manifest_path.name,
        "csv_rows": len(frame),
        "csv_columns": list(frame.columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_path, cif_zip_path, split_path, manifest_path


def verify_outputs(csv_path, cif_zip_path, split_path, target_size):
    frame = pd.read_csv(csv_path)
    if len(frame) != target_size or frame["mof_id"].nunique() != target_size:
        raise RuntimeError("CSV row count or unique-ID count is invalid")
    if frame.isna().any().any():
        raise RuntimeError(
            f"CSV missing columns: {frame.columns[frame.isna().any()].tolist()}"
        )
    co2 = frame[list(CO2_COLUMNS)].to_numpy(float)
    ch4 = frame[list(CH4_COLUMNS)].to_numpy(float)
    if not np.all(co2[:, :-1] <= co2[:, 1:] + 1e-8):
        raise RuntimeError("CO2 monotonicity verification failed")
    if not np.all(ch4[:, :-1] <= ch4[:, 1:] + 1e-8):
        raise RuntimeError("CH4 monotonicity verification failed")

    with zipfile.ZipFile(cif_zip_path) as archive:
        names = archive.namelist()
        if len(names) != target_size or len(set(names)) != target_size:
            raise RuntimeError("CIF ZIP count or uniqueness verification failed")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Corrupt CIF ZIP entry: {corrupt}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_ids = sum(
        [split[f"{name}_mof_ids"] for name in ("train", "val", "test")], []
    )
    if len(split_ids) != target_size or len(set(split_ids)) != target_size:
        raise RuntimeError("Split coverage or uniqueness verification failed")
    if set(split_ids) != set(frame["mof_id"]):
        raise RuntimeError("Split IDs do not match CSV IDs")

    return {
        "csv_rows": len(frame),
        "unique_mof_ids": int(frame["mof_id"].nunique()),
        "cif_files": len(names),
        "missing_cells": int(frame.isna().sum().sum()),
        "co2_monotonic_rows": int(
            np.all(co2[:, :-1] <= co2[:, 1:] + 1e-8, axis=1).sum()
        ),
        "ch4_monotonic_rows": int(
            np.all(ch4[:, :-1] <= ch4[:, 1:] + 1e-8, axis=1).sum()
        ),
        "split_counts": split["counts"],
        "target_ranges_mmol_g": {
            column: [float(frame[column].min()), float(frame[column].max())]
            for column in TARGET_COLUMNS
        },
    }


def main():
    args = parse_args()
    source_zip = args.source_zip or discover_file(SOURCE_ZIP_NAME)
    existing_csv = args.existing_csv or discover_file(EXISTING_CSV_NAME)
    for path in (source_zip, existing_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    with zipfile.ZipFile(source_zip) as archive:
        print("Indexing source archive...", flush=True)
        json_members, cif_members = index_archive(archive)
        print(
            f"Archive index: JSON={len(json_members):,}, CIF={len(cif_members):,}",
            flush=True,
        )
        valid_records, filter_counts = scan_valid_records(
            archive, json_members, cif_members
        )
        valid_by_id = {record["mof_id"]: record for record in valid_records}
        if len(valid_by_id) != len(valid_records):
            raise RuntimeError("Duplicate valid mof_id values")

        existing, existing_ids, retained_ids, rejected_ids, differences = load_existing(
            existing_csv, valid_by_id
        )
        print(
            f"Existing 30K: retained={len(retained_ids):,}, "
            f"replaced={len(rejected_ids):,}",
            flush=True,
        )
        selected, added_ids, selection_stats = select_records(
            valid_records, retained_ids, args.target_size, args.seed
        )
        old_split = reconstruct_old_split(existing, args.seed)
        split_payload = make_fixed_split(
            selected, retained_ids, old_split, args.seed
        )
        manifest = {
            "dataset": "MOFX-DB paired CO2/CH4 adsorption compact 50K",
            "temperature_K": TEMPERATURE_K,
            "adsorption_unit": "mmol/g",
            "unit_note": "1 mol/kg is numerically identical to 1 mmol/g.",
            "random_seed": args.seed,
            "source_archive": str(source_zip),
            "existing_dataset": str(existing_csv),
            "source_archive_counts": {
                "json": len(json_members),
                "cif": len(cif_members),
            },
            "target_pressures_bar": {
                gas: list(spec["pressures"]) for gas, spec in GAS_SPECS.items()
            },
            "target_columns": {
                gas: list(spec["columns"]) for gas, spec in GAS_SPECS.items()
            },
            "filter_counts": dict(filter_counts),
            "filters": [
                "CO2 and CH4 isotherms at 298 K",
                "all five gas-specific target pressures available",
                "adsorption unit compatible with mmol/g",
                "finite nonnegative and monotonic adsorption values",
                "CO2 uptake at 0.5 bar <= 20 mmol/g",
                "finite positive surface area, LCD, and PLD",
                "finite nonnegative void fraction",
                "matching CIF present",
            ],
            "descriptors_retained": [
                "surface_area_m2g",
                "void_fraction",
                "lcd",
                "pld",
                "elements",
            ],
            "existing_30k": {
                "input_rows": len(existing_ids),
                "paired_rows_retained": len(retained_ids),
                "rows_replaced_for_paired_validity": len(rejected_ids),
                "rejected_mof_ids": rejected_ids,
                "co2_target_max_absolute_difference": differences,
            },
            "selection": selection_stats,
            "split": {
                "counts": split_payload["counts"],
                "strategy": split_payload["strategy"],
            },
        }
        csv_path, cif_zip_path, split_path, manifest_path = write_outputs(
            archive,
            selected,
            split_payload,
            args.output_dir,
            manifest,
        )

    verification = verify_outputs(
        csv_path, cif_zip_path, split_path, args.target_size
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification"] = verification
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nBuild complete")
    for label, path in (
        ("CSV", csv_path),
        ("CIF ZIP", cif_zip_path),
        ("Split", split_path),
        ("Manifest", manifest_path),
    ):
        print(f"{label:8s}: {path}")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
