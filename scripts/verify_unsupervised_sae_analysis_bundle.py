from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fast_ai_detector.inference import FastAIDetector  # noqa: E402
from fast_ai_detector.sae_analysis import (  # noqa: E402
    UnsupervisedSAEExplainer,
    load_vector_sae_analysis_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-path",
        default=str(
            ROOT
            / "outputs"
            / "sae_vector_analysis_bundle"
            / "unsupervised_layer17_width_16k_l0_medium"
            / "unsupervised_sae_vector_analysis_bundle_v2.safetensors"
        ),
    )
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
        default=str(ROOT / "outputs" / "sae_vector_analysis_bundle_verification"),
    )
    return parser.parse_args()


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

    texts = (
        pd.read_csv(args.text_csv, usecols=[args.text_column], nrows=args.max_texts)[args.text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    detector = FastAIDetector(mode="unsupervised", device=args.device, load_percentiles=False)
    bundle = load_vector_sae_analysis_bundle(args.bundle_path, device=detector.device, compute_dtype=torch.float32)
    explainer = UnsupervisedSAEExplainer(detector, bundle)

    raw_reps = explainer.predict_raw_representations(texts, batch_size=8)
    runtime_scores = torch.tensor(detector.score_texts(texts, batch_size=8).scores, dtype=torch.float32)
    bundle_scores = explainer.detector_scores_from_raw(raw_reps).cpu()
    state_vs_midpoint, _, ai_net_push = explainer.feature_geometry_from_raw(raw_reps)
    state_vs_midpoint = state_vs_midpoint.cpu()
    ai_net_push = ai_net_push.cpu()

    rows: list[dict[str, object]] = []
    for row_index, text in enumerate(texts):
        row = state_vs_midpoint[row_index]
        push_row = ai_net_push[row_index]
        pos = torch.nonzero(push_row > 0, as_tuple=False).squeeze(-1)
        neg = torch.nonzero(push_row < 0, as_tuple=False).squeeze(-1)
        explanation = explainer.explain_texts([text], batch_size=1, top_k=args.top_k)[0]
        rows.append(
            {
                "row_index": row_index,
                "text_preview": text[:200],
                "runtime_score": float(runtime_scores[row_index].item()),
                "bundle_score": float(bundle_scores[row_index].item()),
                "n_positive_features": int(pos.numel()),
                "n_negative_features": int(neg.numel()),
                "top_explained": len(explanation.top_features),
                "top_abs_state_self_overlap": topk_overlap(torch.abs(row), torch.abs(row), args.top_k, largest=True),
                "top_abs_push_self_overlap": topk_overlap(torch.abs(push_row), torch.abs(push_row), args.top_k, largest=True),
            }
        )

    rows_frame = pd.DataFrame(rows)
    rows_frame.to_csv(output_dir / "per_text_metrics.csv", index=False)

    summary = {
        "bundle_path": str(Path(args.bundle_path).resolve()),
        "text_csv": str(Path(args.text_csv).resolve()),
        "text_column": args.text_column,
        "n_texts": len(texts),
        "verification_scope": [
            "bundle detector-score consistency with runtime unsupervised scoring",
            "non-empty state-vs-midpoint and AI-net-push rankings",
            "explained-feature counts in the emitted top-k lists",
        ],
        "score_max_abs_diff": float((runtime_scores - bundle_scores).abs().max().item()),
        "score_mae": float((runtime_scores - bundle_scores).abs().mean().item()),
        "mean_nonzero_state_features": float((state_vs_midpoint != 0).sum(dim=1).float().mean().item()),
        "mean_nonzero_push_features": float((ai_net_push != 0).sum(dim=1).float().mean().item()),
        "mean_top_explained": float(rows_frame["top_explained"].mean()),
        "top_k": int(args.top_k),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
