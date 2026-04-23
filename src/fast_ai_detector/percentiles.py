from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np


def _assets_root() -> Path:
    return Path(str(resources.files("fast_ai_detector").joinpath("assets")))


def percentile_artifact_path_for_mode(mode: str) -> Path:
    name = f"{mode}_raid_val_balanced_percentiles_v1.npz"
    path = _assets_root() / "percentiles" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing percentile artifact for mode {mode!r}: {path}")
    return path


def default_percentile_grid(num_knots: int = 257, logit_limit: float = 8.0) -> np.ndarray:
    if num_knots < 3:
        raise ValueError("num_knots must be at least 3.")
    t = np.linspace(-logit_limit, logit_limit, int(num_knots), dtype=np.float64)
    grid = 1.0 / (1.0 + np.exp(-t))
    grid[0] = 0.0
    grid[-1] = 1.0
    return np.unique(grid)


def _quantile(scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        return np.quantile(scores, grid, method="linear")
    except TypeError:
        return np.quantile(scores, grid, interpolation="linear")


def _dedupe_score_knots(score_knots: np.ndarray, cdf_knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if score_knots.ndim != 1 or cdf_knots.ndim != 1:
        raise ValueError("Expected 1D knots.")
    if len(score_knots) != len(cdf_knots):
        raise ValueError("Score and CDF knots must have matching length.")
    if len(score_knots) == 0:
        raise ValueError("Knots must be non-empty.")
    keep = np.ones(len(score_knots), dtype=bool)
    keep[:-1] = score_knots[1:] > score_knots[:-1]
    return score_knots[keep], cdf_knots[keep]


def build_class_cdf_knots(scores: np.ndarray, *, grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be 1D.")
    if len(scores) == 0:
        raise ValueError("scores must be non-empty.")
    grid = default_percentile_grid() if grid is None else np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 3:
        raise ValueError("grid must be a 1D array with at least 3 points.")
    if grid[0] < 0 or grid[-1] > 1 or not np.all(grid[1:] >= grid[:-1]):
        raise ValueError("grid must be monotone in [0, 1].")
    score_knots = _quantile(scores, grid)
    score_knots, cdf_knots = _dedupe_score_knots(score_knots.astype(np.float32), grid.astype(np.float32))
    return score_knots, cdf_knots


@dataclass
class PercentileArtifact:
    artifact_version: int
    mode: str
    reference_name: str
    negative_mag_knots: np.ndarray
    negative_mag_cdf_knots: np.ndarray
    positive_score_knots: np.ndarray
    positive_score_cdf_knots: np.ndarray
    n_ai_reference: int
    n_human_reference: int
    metadata: dict[str, Any]

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            artifact_version=np.asarray([self.artifact_version], dtype=np.int32),
            mode=np.asarray([self.mode]),
            reference_name=np.asarray([self.reference_name]),
            negative_mag_knots=self.negative_mag_knots.astype(np.float32),
            negative_mag_cdf_knots=self.negative_mag_cdf_knots.astype(np.float32),
            positive_score_knots=self.positive_score_knots.astype(np.float32),
            positive_score_cdf_knots=self.positive_score_cdf_knots.astype(np.float32),
            n_ai_reference=np.asarray([self.n_ai_reference], dtype=np.int64),
            n_human_reference=np.asarray([self.n_human_reference], dtype=np.int64),
            metadata_json=np.asarray([json.dumps(self.metadata, sort_keys=True)]),
        )


def load_percentile_artifact(path: str | Path) -> PercentileArtifact:
    data = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"][0]))
    return PercentileArtifact(
        artifact_version=int(data["artifact_version"][0]),
        mode=str(data["mode"][0]),
        reference_name=str(data["reference_name"][0]),
        negative_mag_knots=np.asarray(data["negative_mag_knots"], dtype=np.float32),
        negative_mag_cdf_knots=np.asarray(data["negative_mag_cdf_knots"], dtype=np.float32),
        positive_score_knots=np.asarray(data["positive_score_knots"], dtype=np.float32),
        positive_score_cdf_knots=np.asarray(data["positive_score_cdf_knots"], dtype=np.float32),
        n_ai_reference=int(data["n_ai_reference"][0]),
        n_human_reference=int(data["n_human_reference"][0]),
        metadata=metadata,
    )


def load_percentile_artifact_for_mode(mode: str) -> PercentileArtifact:
    return load_percentile_artifact(percentile_artifact_path_for_mode(mode))


def build_percentile_artifact(
    *,
    mode: str,
    reference_name: str,
    ai_scores: np.ndarray,
    human_scores: np.ndarray,
    grid: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> PercentileArtifact:
    ai_scores = np.asarray(ai_scores, dtype=np.float64)
    human_scores = np.asarray(human_scores, dtype=np.float64)
    reference_scores = np.concatenate([human_scores, ai_scores], axis=0)
    negative_mags = -reference_scores[reference_scores < 0]
    positive_scores = reference_scores[reference_scores >= 0]
    if len(negative_mags) == 0 or len(positive_scores) == 0:
        raise ValueError("Reference scores must contain both negative and non-negative values.")
    negative_mag_knots, negative_mag_cdf_knots = build_class_cdf_knots(negative_mags, grid=grid)
    positive_score_knots, positive_score_cdf_knots = build_class_cdf_knots(positive_scores, grid=grid)
    return PercentileArtifact(
        artifact_version=1,
        mode=mode,
        reference_name=reference_name,
        negative_mag_knots=negative_mag_knots,
        negative_mag_cdf_knots=negative_mag_cdf_knots,
        positive_score_knots=positive_score_knots,
        positive_score_cdf_knots=positive_score_cdf_knots,
        n_ai_reference=int(len(ai_scores)),
        n_human_reference=int(len(human_scores)),
        metadata={} if metadata is None else dict(metadata),
    )


def cdf_from_knots(scores: np.ndarray | float, score_knots: np.ndarray, cdf_knots: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.interp(values, score_knots.astype(np.float64), cdf_knots.astype(np.float64), left=0.0, right=1.0)


def score_percentiles(
    scores: np.ndarray | float,
    artifact: PercentileArtifact,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    pct = np.empty_like(values, dtype=np.float64)
    nonneg = values >= 0
    if np.any(nonneg):
        pos_cdf = cdf_from_knots(values[nonneg], artifact.positive_score_knots, artifact.positive_score_cdf_knots)
        pct[nonneg] = 50.0 + 50.0 * pos_cdf
    if np.any(~nonneg):
        neg_cdf = cdf_from_knots(-values[~nonneg], artifact.negative_mag_knots, artifact.negative_mag_cdf_knots)
        pct[~nonneg] = 50.0 * (1.0 - neg_cdf)
    return pct


def empirical_percentiles(scores: np.ndarray | float, reference_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ref = np.sort(np.asarray(reference_scores, dtype=np.float64))
    cdf = np.searchsorted(ref, values, side="right") / max(1, len(ref))
    return 100.0 * cdf
