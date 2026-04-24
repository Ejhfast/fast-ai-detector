from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class FeatureLookupRow:
    feature_index: int
    status: str
    neuronpedia_title: str
    neuronpedia_source: str
    neuronpedia_url: str


def _assets_root() -> Path:
    return Path(str(resources.files("fast_ai_detector").joinpath("assets")))


def default_feature_lookup_path() -> Path:
    return _assets_root() / "feature_lookup" / "layer_17_width_16k_l0_medium_neuronpedia_lookup_v1.csv.gz"


def load_feature_lookup(path: str | Path | None = None) -> dict[int, FeatureLookupRow]:
    lookup_path = default_feature_lookup_path() if path is None else Path(path)
    rows: dict[int, FeatureLookupRow] = {}
    with gzip.open(lookup_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature_index = int(row["feature_index"])
            rows[feature_index] = FeatureLookupRow(
                feature_index=feature_index,
                status=row.get("status", ""),
                neuronpedia_title=row.get("neuronpedia_title", ""),
                neuronpedia_source=row.get("neuronpedia_source", ""),
                neuronpedia_url=row.get("neuronpedia_url", ""),
            )
    return rows
