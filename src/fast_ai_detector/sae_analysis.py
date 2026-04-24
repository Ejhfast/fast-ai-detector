from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

from .feature_lookup import FeatureLookupRow, load_feature_lookup
from .inference import FastAIDetector
from .models import UnsupervisedState, _download_hf_file, load_manifest, tokenize_batch


@dataclass
class ReducedSAEAnalysisBundle:
    tensor_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    w_enc: torch.Tensor
    b_enc: torch.Tensor
    threshold: torch.Tensor
    raw_readout: torch.Tensor
    feature_alignment: torch.Tensor
    raw_intercept: float
    sae_score_offset: float

    @property
    def d_in(self) -> int:
        return int(self.w_enc.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.w_enc.shape[1])


@dataclass
class SAEFeatureContribution:
    feature_index: int
    title: str
    contribution: float
    share_of_side_pct: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "feature_index": self.feature_index,
            "title": self.title,
            "contribution": self.contribution,
            "share_of_side_pct": self.share_of_side_pct,
        }


@dataclass
class SAEExplanation:
    detector_score: float
    total_active_features: int
    top_ai_features: list[SAEFeatureContribution]
    top_human_features: list[SAEFeatureContribution]

    def to_dict(self) -> dict[str, object]:
        return {
            "detector_score": self.detector_score,
            "total_active_features": self.total_active_features,
            "top_ai_features": [item.to_dict() for item in self.top_ai_features],
            "top_human_features": [item.to_dict() for item in self.top_human_features],
            "note": (
                "Heuristic pooled-SAE interpretation on the student-predicted document vector. "
                "This view shows explained Neuronpedia features only and is not an exact detector-score decomposition."
            ),
        }


def _metadata_path_for_bundle(bundle_path: Path) -> Path:
    return bundle_path.with_suffix(".json")


def resolve_default_reduced_sae_analysis_bundle_paths() -> tuple[Path, Path]:
    manifest = load_manifest()
    tensor_path = _download_hf_file(dict(manifest["unsupervised_sae_analysis_bundle"]))
    metadata_path = _download_hf_file(dict(manifest["unsupervised_sae_analysis_metadata"]))
    return tensor_path, metadata_path


def load_reduced_sae_analysis_bundle(
    bundle_path: str | Path | None = None,
    *,
    metadata_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> ReducedSAEAnalysisBundle:
    if bundle_path is None:
        resolved_tensor_path, resolved_metadata_path = resolve_default_reduced_sae_analysis_bundle_paths()
        path = resolved_tensor_path
        metadata_path_obj = resolved_metadata_path if metadata_path is None else Path(metadata_path)
    else:
        path = Path(bundle_path)
        metadata_path_obj = _metadata_path_for_bundle(path) if metadata_path is None else Path(metadata_path)
    metadata = json.loads(metadata_path_obj.read_text(encoding="utf-8"))
    tensors = load_file(str(path), device=str(torch.device(device)))
    return ReducedSAEAnalysisBundle(
        tensor_path=path,
        metadata_path=metadata_path_obj,
        metadata=metadata,
        w_enc=tensors["W_enc"].to(device=device, dtype=compute_dtype),
        b_enc=tensors["b_enc"].to(device=device, dtype=compute_dtype),
        threshold=tensors["threshold"].to(device=device, dtype=compute_dtype),
        raw_readout=tensors["raw_readout"].to(device=device, dtype=compute_dtype),
        feature_alignment=tensors["feature_alignment"].to(device=device, dtype=compute_dtype),
        raw_intercept=float(tensors["raw_intercept"].item()),
        sae_score_offset=float(tensors["sae_score_offset"].item()),
    )


class UnsupervisedSAEExplainer:
    def __init__(
        self,
        detector: FastAIDetector,
        bundle: ReducedSAEAnalysisBundle,
    ):
        if detector.mode != "unsupervised":
            raise ValueError("SAE analysis is only supported for unsupervised mode.")
        state = detector.state
        if not isinstance(state, UnsupervisedState):
            raise TypeError(f"Expected UnsupervisedState, got {type(state)!r}")
        self.detector = detector
        self.state = state
        self.bundle = bundle
        if bundle.d_in != int(state.student.target_mean.numel()):
            raise ValueError(
                f"Bundle input dim {bundle.d_in} does not match student output dim "
                f"{int(state.student.target_mean.numel())}."
            )
        self._feature_lookup: dict[int, FeatureLookupRow] | None = None

    def predict_raw_representations(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> torch.Tensor:
        clean_texts = ["" if text is None else str(text) for text in texts]
        student = self.state.student
        chunks: list[torch.Tensor] = []
        use_amp = student.device.type == "cuda"
        with torch.no_grad():
            for start in range(0, len(clean_texts), batch_size):
                batch_texts = clean_texts[start : start + batch_size]
                tokens = tokenize_batch(student.tokenizer, batch_texts, student.max_length, student.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred_z = student.model(tokens["input_ids"], tokens["attention_mask"])
                    pred_raw = pred_z * student.target_scale + student.target_mean
                chunks.append(pred_raw.float())
        if not chunks:
            return torch.empty((0, self.bundle.d_in), device=student.device, dtype=torch.float32)
        return torch.cat(chunks, dim=0)

    def encode_raw_representations(self, raw_reps: torch.Tensor) -> torch.Tensor:
        bundle = self.bundle
        hidden_pre = raw_reps.to(bundle.w_enc.dtype) @ bundle.w_enc + bundle.b_enc
        return torch.where(hidden_pre > bundle.threshold, hidden_pre, torch.zeros_like(hidden_pre))

    def detector_scores_from_raw(self, raw_reps: torch.Tensor) -> torch.Tensor:
        return raw_reps.to(self.bundle.raw_readout.dtype) @ self.bundle.raw_readout + self.bundle.raw_intercept

    def feature_contributions_from_raw(self, raw_reps: torch.Tensor) -> torch.Tensor:
        feature_acts = self.encode_raw_representations(raw_reps)
        return feature_acts * self.bundle.feature_alignment

    def feature_lookup(self) -> dict[int, FeatureLookupRow]:
        if self._feature_lookup is None:
            self._feature_lookup = load_feature_lookup()
        return self._feature_lookup

    def _build_feature_items(
        self,
        *,
        contributions: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
        side_total_abs: float,
    ) -> list[SAEFeatureContribution]:
        lookup = self.feature_lookup()
        items: list[SAEFeatureContribution] = []
        for feature_index in indices.tolist():
            contribution = float(contributions[feature_index].item())
            meta = lookup.get(
                int(feature_index),
                FeatureLookupRow(
                    feature_index=int(feature_index),
                    status="missing_lookup",
                    neuronpedia_title="",
                    neuronpedia_source="",
                    neuronpedia_url="",
                ),
            )
            if meta.status != "ok" or not meta.neuronpedia_title or meta.neuronpedia_title == "NO_EXPLANATION":
                continue
            items.append(
                SAEFeatureContribution(
                    feature_index=int(feature_index),
                    title=meta.neuronpedia_title,
                    contribution=contribution,
                    share_of_side_pct=0.0 if side_total_abs == 0 else 100.0 * abs(contribution) / side_total_abs,
                )
            )
            if len(items) >= top_k:
                break
        return items

    @staticmethod
    def _sorted_contributing_indices(row: torch.Tensor, *, positive: bool) -> torch.Tensor:
        if positive:
            candidate_indices = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
            if candidate_indices.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=row.device)
            candidate_values = row[candidate_indices]
            order = torch.argsort(candidate_values, descending=True)
            return candidate_indices[order]
        candidate_indices = torch.nonzero(row < 0, as_tuple=False).squeeze(-1)
        if candidate_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=row.device)
        candidate_values = row[candidate_indices]
        order = torch.argsort(candidate_values, descending=False)
        return candidate_indices[order]

    def sae_scores_from_raw(self, raw_reps: torch.Tensor) -> torch.Tensor:
        contributions = self.feature_contributions_from_raw(raw_reps)
        return contributions.sum(dim=-1) + self.bundle.sae_score_offset

    def explain_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        top_k: int = 10,
    ) -> list[SAEExplanation]:
        raw_reps = self.predict_raw_representations(texts, batch_size=batch_size)
        detector_scores = self.detector_scores_from_raw(raw_reps)
        feature_acts = self.encode_raw_representations(raw_reps)
        contributions = feature_acts * self.bundle.feature_alignment
        explanations: list[SAEExplanation] = []
        for index in range(len(texts)):
            row = contributions[index]
            acts = feature_acts[index]
            top_positive_indices = self._sorted_contributing_indices(row, positive=True)
            top_negative_indices = self._sorted_contributing_indices(row, positive=False)
            positive_total_abs = float(row[row > 0].abs().sum().item())
            negative_total_abs = float(row[row < 0].abs().sum().item())
            top_ai_features = self._build_feature_items(
                contributions=row,
                indices=top_positive_indices,
                top_k=top_k,
                side_total_abs=positive_total_abs,
            )
            top_human_features = self._build_feature_items(
                contributions=row,
                indices=top_negative_indices,
                top_k=top_k,
                side_total_abs=negative_total_abs,
            )
            explanations.append(
                SAEExplanation(
                    detector_score=float(detector_scores[index].item()),
                    total_active_features=int((acts > 0).sum().item()),
                    top_ai_features=top_ai_features,
                    top_human_features=top_human_features,
                )
            )
        return explanations
