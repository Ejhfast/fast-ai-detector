from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fast_ai_detector.inference import FastAIDetector  # noqa: E402
from fast_ai_detector.sae_analysis import (  # noqa: E402
    UnsupervisedSAEExplainer,
    load_reduced_sae_analysis_bundle,
)


LOCAL_RELEASE_PATHS = {
    "gemma-scope-2-4b-pt-res-all": (
        ROOT.parent
        / "replicate_gemma_4_13"
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--google--gemma-scope-2-4b-pt"
        / "snapshots"
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-path",
        default=str(
            ROOT
            / "outputs"
            / "sae_analysis_bundle"
            / "unsupervised_layer17_width_16k_l0_medium"
            / "unsupervised_sae_analysis_bundle_v1.safetensors"
        ),
    )
    parser.add_argument("--sae-release", default="gemma-scope-2-4b-pt-res-all")
    parser.add_argument("--sae-id", default="layer_17_width_16k_l0_medium")
    parser.add_argument(
        "--text-csv",
        default=str(ROOT.parent / "data" / "raid" / "train_shuffled_seed0.csv"),
    )
    parser.add_argument("--text-column", default="generation")
    parser.add_argument("--max-texts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "sae_analysis_bundle_verification"),
    )
    return parser.parse_args()


def resolve_local_sae_dir(release: str, sae_id: str) -> Path:
    if release not in LOCAL_RELEASE_PATHS:
        raise FileNotFoundError(f"Unknown local SAE release {release!r}")
    snapshots_root = LOCAL_RELEASE_PATHS[release]
    matches = sorted(snapshots_root.glob(f"*/resid_post_all/{sae_id}"))
    if not matches:
        matches = sorted(snapshots_root.glob(f"*/resid_post/{sae_id}"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find cached SAE for release={release!r} sae_id={sae_id!r} under {snapshots_root}"
        )
    return matches[-1]


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int, *, largest: bool) -> float:
    k = min(k, int(a.numel()), int(b.numel()))
    a_idx = set(torch.topk(a, k=k, largest=largest).indices.tolist())
    b_idx = set(torch.topk(b, k=k, largest=largest).indices.tolist())
    if not a_idx:
        return 1.0
    return float(len(a_idx & b_idx) / len(a_idx | b_idx))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts_frame = pd.read_csv(args.text_csv, usecols=[args.text_column], nrows=args.max_texts)
    texts = texts_frame[args.text_column].fillna("").astype(str).tolist()

    detector = FastAIDetector(mode="unsupervised", device=args.device, load_percentiles=False)
    bundle = load_reduced_sae_analysis_bundle(args.bundle_path, device=detector.device, compute_dtype=torch.float32)
    explainer = UnsupervisedSAEExplainer(detector, bundle)

    raw_reps = explainer.predict_raw_representations(texts, batch_size=8)
    detector_scores_from_bundle = explainer.detector_scores_from_raw(raw_reps).cpu()
    detector_scores_runtime = torch.tensor(
        detector.score_texts(texts, batch_size=8).scores,
        dtype=torch.float32,
    )

    sae_dir = resolve_local_sae_dir(args.sae_release, args.sae_id)
    full_tensors = load_file(str(sae_dir / "params.safetensors"), device=str(detector.device))
    w_enc_full = full_tensors["w_enc"].float()
    b_enc_full = full_tensors["b_enc"].float()
    threshold_full = full_tensors["threshold"].float()
    w_dec_full = full_tensors["w_dec"].float()

    raw_readout = bundle.raw_readout.float()
    feature_alignment_full = (w_dec_full @ raw_readout).cpu()

    full_hidden = raw_reps.float() @ w_enc_full + b_enc_full
    full_features = torch.where(full_hidden > threshold_full, full_hidden, torch.zeros_like(full_hidden))
    full_contributions = full_features * feature_alignment_full.to(full_features.device)
    reduced_features = explainer.encode_raw_representations(raw_reps).cpu()
    reduced_contributions = (reduced_features * bundle.feature_alignment.cpu()).cpu()

    rows: list[dict[str, object]] = []
    pos_overlaps: list[float] = []
    neg_overlaps: list[float] = []
    for idx, text in enumerate(texts):
        pos_overlap = topk_overlap(full_contributions[idx], reduced_contributions[idx], args.top_k, largest=True)
        neg_overlap = topk_overlap(full_contributions[idx], reduced_contributions[idx], args.top_k, largest=False)
        pos_overlaps.append(pos_overlap)
        neg_overlaps.append(neg_overlap)
        rows.append(
            {
                "row_index": idx,
                "text_preview": text[:200],
                "detector_score_runtime": float(detector_scores_runtime[idx].item()),
                "detector_score_from_bundle": float(detector_scores_from_bundle[idx].item()),
                "feature_mae": float((reduced_features[idx] - full_features[idx].cpu()).abs().mean().item()),
                "contribution_mae": float(
                    (reduced_contributions[idx] - full_contributions[idx].cpu()).abs().mean().item()
                ),
                "top_positive_jaccard": pos_overlap,
                "top_negative_jaccard": neg_overlap,
            }
        )

    rows_frame = pd.DataFrame(rows)
    rows_frame.to_csv(output_dir / "per_text_metrics.csv", index=False)

    summary = {
        "bundle_path": str(Path(args.bundle_path).resolve()),
        "sae_dir": str(sae_dir.resolve()),
        "text_csv": str(Path(args.text_csv).resolve()),
        "text_column": args.text_column,
        "n_texts": len(texts),
        "verification_scope": [
            "detector raw-score consistency between runtime and bundle readout",
            "reduced bundle vs full SAE feature activations on the same pooled/student vector",
            "reduced bundle vs full SAE top positive/negative contribution ranking overlap",
        ],
        "detector_runtime_vs_bundle_max_abs_diff": float(
            (detector_scores_runtime - detector_scores_from_bundle).abs().max().item()
        ),
        "detector_runtime_vs_bundle_mae": float(
            (detector_scores_runtime - detector_scores_from_bundle).abs().mean().item()
        ),
        "full_vs_reduced_feature_mae": float((reduced_features - full_features.cpu()).abs().mean().item()),
        "full_vs_reduced_feature_max_abs_diff": float(
            (reduced_features - full_features.cpu()).abs().max().item()
        ),
        "full_vs_reduced_contribution_mae": float(
            (reduced_contributions - full_contributions.cpu()).abs().mean().item()
        ),
        "full_vs_reduced_contribution_max_abs_diff": float(
            (reduced_contributions - full_contributions.cpu()).abs().max().item()
        ),
        "mean_top_positive_jaccard": float(sum(pos_overlaps) / len(pos_overlaps)),
        "mean_top_negative_jaccard": float(sum(neg_overlaps) / len(neg_overlaps)),
        "top_k": int(args.top_k),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
