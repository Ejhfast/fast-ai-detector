from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fast_ai_detector.inference import FastAIDetector
from fast_ai_detector.percentiles import (
    build_percentile_artifact,
    default_percentile_grid,
    empirical_percentiles,
    score_percentiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact percentile artifacts for fast-ai-detector from a balanced "
            "reference subset of the deterministic RAID validation split."
        )
    )
    parser.add_argument(
        "--raid-csv",
        default="data/raid/train_shuffled_seed0.csv",
        help="RAID CSV used for the detector training split.",
    )
    parser.add_argument(
        "--student-metadata-csv",
        default=(
            "replicate_gemma_4_13/outputs/feature_cache/"
            "student_raid_train_shuffled_seed0_1024student_final_full/metadata.csv"
        ),
    )
    parser.add_argument(
        "--student-feature-file",
        default=(
            "replicate_gemma_4_13/outputs/feature_cache/"
            "student_raid_train_shuffled_seed0_1024student_final_full/student_layer17_mean_resid_fp32.npy"
        ),
    )
    parser.add_argument(
        "--unsupervised-delta-file",
        default="fast-ai-detector/src/fast_ai_detector/assets/models/raid_unsupervised_delta_v1.npz",
    )
    parser.add_argument("--text-col", default="generation")
    parser.add_argument("--csv-chunksize", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--num-knots", type=int, default=257)
    parser.add_argument("--logit-limit", type=float, default=8.0)
    parser.add_argument(
        "--assets-dir",
        default="fast-ai-detector/src/fast_ai_detector/assets/percentiles",
    )
    parser.add_argument(
        "--output-dir",
        default="fast-ai-detector/outputs/raid_percentiles",
    )
    return parser.parse_args()


def splitmix64_uniform(row_numbers: np.ndarray, seed: int, salt: int) -> np.ndarray:
    x = row_numbers.astype(np.uint64, copy=False)
    x = x + np.uint64(seed + 1) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return ((x >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))


def select_balanced_reference_rows(
    *,
    metadata_csv: Path,
    sample_fraction: float,
    test_size: float,
    random_state: int,
    reference_seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(
        metadata_csv,
        usecols=["feature_row", "_row_number", "label_ai"],
        dtype={"feature_row": np.int64, "_row_number": np.int64, "label_ai": np.int8},
        low_memory=False,
    )
    rows = frame["_row_number"].to_numpy(dtype=np.uint64)
    sample_mask = splitmix64_uniform(rows, random_state, salt=17) < sample_fraction
    split_u = splitmix64_uniform(rows, random_state, salt=23)
    val_mask = sample_mask & (split_u < test_size)
    val = frame.loc[val_mask].copy().reset_index(drop=True)
    humans = val.loc[val["label_ai"] == 0].copy()
    ai = val.loc[val["label_ai"] == 1].copy()
    if humans.empty or ai.empty:
        raise ValueError("Expected both human and AI rows in the RAID validation split.")
    if len(ai) < len(humans):
        raise ValueError(f"Need at least {len(humans)} AI val rows, found {len(ai)}.")
    ai_sample = ai.sample(n=len(humans), random_state=reference_seed).copy()
    reference = (
        pd.concat([humans, ai_sample], ignore_index=True)
        .sort_values("_row_number")
        .reset_index(drop=True)
    )
    counts = {
        "n_val_total": int(len(val)),
        "n_val_ai": int(len(ai)),
        "n_val_human": int(len(humans)),
        "n_reference_total": int(len(reference)),
        "n_reference_ai": int((reference["label_ai"] == 1).sum()),
        "n_reference_human": int((reference["label_ai"] == 0).sum()),
    }
    return reference, counts


def load_reference_texts(
    *,
    raid_csv: Path,
    selected_rows: np.ndarray,
    text_col: str,
    chunksize: int,
) -> pd.DataFrame:
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    parts: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(
        raid_csv,
        usecols=[text_col],
        chunksize=chunksize,
        dtype="string",
    ):
        n = len(chunk)
        rows = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
        mask = np.isin(rows, selected_rows, assume_unique=False)
        if not mask.any():
            continue
        selected = chunk.loc[mask, [text_col]].copy()
        selected["_row_number"] = rows[mask]
        parts.append(selected)
    if not parts:
        raise ValueError("Failed to recover any text rows for the selected reference subset.")
    return pd.concat(parts, ignore_index=True)


def score_unsupervised_from_cache(reference: pd.DataFrame, feature_file: Path, delta_file: Path) -> np.ndarray:
    features = np.load(feature_file, mmap_mode="r")
    feature_rows = reference["feature_row"].to_numpy(dtype=np.int64)
    x = np.asarray(features[feature_rows, 0, :], dtype=np.float32)
    delta = np.load(delta_file)
    mean = np.asarray(delta["mean"], dtype=np.float32)
    scale = np.asarray(delta["scale"], dtype=np.float32)
    direction = np.asarray(delta["direction"], dtype=np.float32)
    intercept = float(np.asarray(delta["intercept"]).reshape(-1)[0])
    z = (x - mean) / scale
    return np.asarray(z @ direction + intercept, dtype=np.float32)


def approximation_summary(
    *,
    scores: np.ndarray,
    artifact,
) -> dict[str, float]:
    approx_pct = score_percentiles(scores, artifact)
    exact_pct = np.empty_like(approx_pct)
    nonneg = scores >= 0
    if np.any(nonneg):
        exact_pct[nonneg] = 50.0 + 0.5 * empirical_percentiles(scores[nonneg], scores[nonneg])
    if np.any(~nonneg):
        exact_pct[~nonneg] = 50.0 * (1.0 - 0.01 * empirical_percentiles(-scores[~nonneg], -scores[~nonneg]))
    err = np.abs(approx_pct - exact_pct)
    return {
        "pct_mae": float(err.mean()),
        "pct_p95_abs_err": float(np.quantile(err, 0.95)),
        "pct_max_abs_err": float(err.max()),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mode_artifact(
    *,
    mode: str,
    score_col: str,
    reference: pd.DataFrame,
    grid: np.ndarray,
    assets_dir: Path,
    build_dir: Path,
    build_metadata: dict[str, object],
) -> dict[str, object]:
    ai_scores = reference.loc[reference["label_ai"] == 1, score_col].to_numpy(dtype=np.float64)
    human_scores = reference.loc[reference["label_ai"] == 0, score_col].to_numpy(dtype=np.float64)
    artifact = build_percentile_artifact(
        mode=mode,
        reference_name="raid_val_balanced_seed0",
        ai_scores=ai_scores,
        human_scores=human_scores,
        grid=grid,
        metadata=build_metadata,
    )
    artifact_path = assets_dir / f"{mode}_raid_val_balanced_percentiles_v1.npz"
    artifact.save_npz(artifact_path)
    summary = {
        "mode": mode,
        "artifact_path": str(artifact_path.resolve()),
        "reference_name": artifact.reference_name,
        "n_ai_reference": artifact.n_ai_reference,
        "n_human_reference": artifact.n_human_reference,
        "n_negative_knots": int(len(artifact.negative_mag_knots)),
        "n_positive_knots": int(len(artifact.positive_score_knots)),
        "score_min": float(reference[score_col].min()),
        "score_max": float(reference[score_col].max()),
        "score_mean_ai": float(ai_scores.mean()),
        "score_mean_human": float(human_scores.mean()),
        "approximation": approximation_summary(
            scores=reference[score_col].to_numpy(dtype=np.float64),
            artifact=artifact,
        ),
        "metadata": build_metadata,
    }
    summary_path = assets_dir / f"{mode}_raid_val_balanced_percentiles_v1.json"
    write_json(summary_path, summary)
    write_json(build_dir / f"{mode}_artifact_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    assets_dir = Path(args.assets_dir)
    output_dir = Path(args.output_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, counts = select_balanced_reference_rows(
        metadata_csv=Path(args.student_metadata_csv),
        sample_fraction=float(args.sample_fraction),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        reference_seed=int(args.reference_seed),
    )
    texts = load_reference_texts(
        raid_csv=Path(args.raid_csv),
        selected_rows=reference["_row_number"].to_numpy(dtype=np.int64),
        text_col=args.text_col,
        chunksize=int(args.csv_chunksize),
    )
    reference = reference.merge(texts, on="_row_number", how="left", validate="one_to_one")
    if reference[args.text_col].isna().any():
        missing = int(reference[args.text_col].isna().sum())
        raise ValueError(f"Missing text for {missing} selected reference rows.")

    reference["unsupervised_score"] = score_unsupervised_from_cache(
        reference,
        feature_file=Path(args.student_feature_file),
        delta_file=Path(args.unsupervised_delta_file),
    )
    detector = FastAIDetector(mode="raid-finetune", device=args.device, load_percentiles=False)
    finetune_scores = detector.score_texts(
        reference[args.text_col].fillna("").astype(str).tolist(),
        batch_size=int(args.batch_size),
    ).scores
    reference["raid_finetune_score"] = np.asarray(finetune_scores, dtype=np.float32)

    reference_csv = output_dir / "raid_val_balanced_reference_scores.csv.gz"
    reference.drop(columns=[args.text_col]).to_csv(reference_csv, index=False, compression="gzip")

    grid = default_percentile_grid(num_knots=int(args.num_knots), logit_limit=float(args.logit_limit))
    build_metadata = {
        "csv_path": str(Path(args.raid_csv).resolve()),
        "student_metadata_csv": str(Path(args.student_metadata_csv).resolve()),
        "student_feature_file": str(Path(args.student_feature_file).resolve()),
        "unsupervised_delta_file": str(Path(args.unsupervised_delta_file).resolve()),
        "text_col": args.text_col,
        "sample_fraction": float(args.sample_fraction),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "reference_seed": int(args.reference_seed),
        "num_knots_requested": int(args.num_knots),
        "logit_limit": float(args.logit_limit),
        **counts,
    }
    unsup_summary = build_mode_artifact(
        mode="unsupervised",
        score_col="unsupervised_score",
        reference=reference,
        grid=grid,
        assets_dir=assets_dir,
        build_dir=output_dir,
        build_metadata=build_metadata,
    )
    finetune_summary = build_mode_artifact(
        mode="raid-finetune",
        score_col="raid_finetune_score",
        reference=reference,
        grid=grid,
        assets_dir=assets_dir,
        build_dir=output_dir,
        build_metadata=build_metadata,
    )

    summary = {
        "reference_csv": str(reference_csv.resolve()),
        "assets_dir": str(assets_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "reference_counts": counts,
        "modes": {
            "unsupervised": unsup_summary,
            "raid-finetune": finetune_summary,
        },
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
