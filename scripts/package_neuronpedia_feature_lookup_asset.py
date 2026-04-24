from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


CSV_COLUMNS = [
    "feature_index",
    "status",
    "neuronpedia_title",
    "neuronpedia_source",
    "neuronpedia_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=(
            "fast-ai-detector/outputs/neuronpedia_lookup/"
            "gemma_3_4b__17_gemmascope_2_res_16k__feature_lookup.csv"
        ),
    )
    parser.add_argument(
        "--output-csv-gz",
        default=(
            "fast-ai-detector/src/fast_ai_detector/assets/feature_lookup/"
            "layer_17_width_16k_l0_medium_neuronpedia_lookup_v1.csv.gz"
        ),
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv_gz = Path(args.output_csv_gz)
    output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else output_csv_gz.with_suffix("").with_suffix(".json")
    )

    rows: list[dict[str, str]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "feature_index": str(int(row["feature_index"])),
                    "status": row.get("status", ""),
                    "neuronpedia_title": row.get("neuronpedia_title", ""),
                    "neuronpedia_source": row.get("neuronpedia_source", ""),
                    "neuronpedia_url": row.get("neuronpedia_url", ""),
                }
            )
    rows.sort(key=lambda row: int(row["feature_index"]))

    with gzip.open(output_csv_gz, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "input_csv": str(input_csv.resolve()),
        "output_csv_gz": str(output_csv_gz.resolve()),
        "num_rows": len(rows),
        "status_counts": status_counts,
        "columns": CSV_COLUMNS,
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
