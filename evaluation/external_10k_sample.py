#!/usr/bin/env python3
"""Create a leakage-free external hMOF candidate pool from the MOFX-DB archive.

The first 10,000 rows are the prespecified evaluation sample.  A further 500
rows are prespecified reserves, used in order only if CIF parsing or feature
construction fails.  Targets are written to a separate file and are not needed
by the inference program.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ZIP = REPO_ROOT / "data" / "raw" / "bulk-dl-mofdb-version-dc8a0295db.zip"
DEFAULT_EXISTING_CSV = REPO_ROOT / "data" / "processed" / "MOF_CO2_CH4_50K.csv"
DEFAULT_BUILDER = REPO_ROOT / "pipeline" / "13.Build_MOF_CO2_CH4_50K.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "external_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE_ZIP)
    parser.add_argument("--existing-csv", type=Path, default=DEFAULT_EXISTING_CSV)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--reserve-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser.parse_args()


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("mof50k_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import dataset builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.source_zip, args.existing_csv, args.builder):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.sample_size <= 0 or args.reserve_size < 0:
        raise ValueError("sample-size must be positive and reserve-size non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    builder = load_builder(args.builder)

    existing = pd.read_csv(args.existing_csv, usecols=["mof_id"])
    existing_ids = {
        builder.canonical_mof_id(value) for value in existing["mof_id"].astype(str)
    }
    if None in existing_ids or len(existing_ids) != len(existing):
        raise RuntimeError("The existing 50K CSV has invalid or duplicate mof_id values")

    with zipfile.ZipFile(args.source_zip) as archive:
        json_members, cif_members = builder.index_archive(archive)
        valid_records, status_counts = builder.scan_valid_records(
            archive, json_members, cif_members
        )
        valid_by_id = {row["mof_id"]: row for row in valid_records}
        remaining_ids = sorted(
            set(valid_by_id) - existing_ids,
            key=lambda value: int(value.split("-")[1]),
        )
        requested = args.sample_size + args.reserve_size
        if len(remaining_ids) < requested:
            raise RuntimeError(
                f"Need {requested:,} candidates but only {len(remaining_ids):,} remain"
            )

        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(remaining_ids))[:requested]
        selected_ids = [remaining_ids[int(index)] for index in order]
        selected = [valid_by_id[mof_id] for mof_id in selected_ids]

        rows = []
        for rank, row in enumerate(selected):
            clean = {key: value for key, value in row.items() if key != "_cif_member"}
            clean["selection_rank"] = rank
            clean["selection_role"] = (
                "primary" if rank < args.sample_size else "reserve"
            )
            rows.append(clean)
        frame = pd.DataFrame(rows)

        target_columns = list(builder.TARGET_COLUMNS)
        input_columns = [
            "selection_rank",
            "selection_role",
            "mof_id",
            "surface_area_m2g",
            "void_fraction",
            "lcd",
            "pld",
            "elements",
            "has_cif",
        ]
        input_path = args.output_dir / "external_candidate_pool_inputs.csv"
        label_path = args.output_dir / "external_candidate_pool_labels_private.csv"
        cif_path = args.output_dir / "external_candidate_pool_cifs.zip"
        manifest_path = args.output_dir / "external_10k_sampling_manifest.json"

        frame[input_columns].to_csv(input_path, index=False)
        frame[["selection_rank", "selection_role", "mof_id", *target_columns]].to_csv(
            label_path, index=False
        )

        with zipfile.ZipFile(
            cif_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as output_zip:
            for row in selected:
                output_zip.writestr(
                    f"{row['mof_id']}.cif", archive.read(row["_cif_member"])
                )

    manifest = {
        "purpose": "External inference evaluation with no overlap with the full original 50K",
        "sampling_method": (
            "Uniform random sampling without replacement from sorted valid remaining "
            "MOFX-DB records; NumPy PCG64 generator. The primary 10K and ordered "
            "reserve are fixed before CIF parsing and inference."
        ),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "reserve_size": args.reserve_size,
        "source_zip": str(args.source_zip),
        "source_zip_sha256": sha256_file(args.source_zip),
        "existing_csv": str(args.existing_csv),
        "existing_csv_sha256": sha256_file(args.existing_csv),
        "existing_id_count": len(existing_ids),
        "valid_population_count": len(valid_records),
        "remaining_valid_population_count": len(remaining_ids),
        "archive_status_counts": dict(sorted(status_counts.items())),
        "selected_id_order_sha256": sha256_lines(selected_ids),
        "primary_id_order_sha256": sha256_lines(selected_ids[: args.sample_size]),
        "files": {
            input_path.name: sha256_file(input_path),
            label_path.name: sha256_file(label_path),
            cif_path.name: sha256_file(cif_path),
        },
        "target_columns": target_columns,
        "label_separation_note": (
            "The inference pipeline must read only the inputs CSV and CIF ZIP. "
            "The private labels CSV is opened only after predictions are complete."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    primary = set(selected_ids[: args.sample_size])
    reserve = set(selected_ids[args.sample_size :])
    if primary & existing_ids or reserve & existing_ids or primary & reserve:
        raise RuntimeError("Leakage or duplicate selection detected")
    print(f"Valid population: {len(valid_records):,}")
    print(f"Excluded original set: {len(existing_ids):,}")
    print(f"Remaining valid population: {len(remaining_ids):,}")
    print(f"Primary/reserve selected: {len(primary):,}/{len(reserve):,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
