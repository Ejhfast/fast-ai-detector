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
class VectorSAEAnalysisBundle:
    tensor_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    w_dec: torch.Tensor
    transform_mean: torch.Tensor
    transform_scale: torch.Tensor
    direction: torch.Tensor
    midpoint_z: torch.Tensor
    intercept: float

    @property
    def d_in(self) -> int:
        return int(self.w_dec.shape[1])

    @property
    def d_sae(self) -> int:
        return int(self.w_dec.shape[0])


@dataclass
class SAEFeatureContribution:
    feature_index: int
    title: str
    state_vs_midpoint: float
    usual_association: str
    ai_net_push: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "feature_index": self.feature_index,
            "title": self.title,
            "state_vs_midpoint": self.state_vs_midpoint,
            "usual_association": self.usual_association,
            "ai_net_push": self.ai_net_push,
        }


def _note_text() -> str:
    return (
        "Vector-space SAE interpretation of the example's position relative to the RAID midpoint in SAE space. "
        "The surfaced list keeps only positive midpoint loadings, so it reads as features the document has more of than "
        "the RAID midpoint. Each feature is annotated with its usual detector association and AI-direction net push, "
        "shown for explained Neuronpedia features only."
    )


@dataclass
class SAEExplanation:
    detector_score: float
    total_ranked_features: int
    top_features: list[SAEFeatureContribution]

    def to_dict(self) -> dict[str, object]:
        return {
            "detector_score": self.detector_score,
            "total_ranked_features": self.total_ranked_features,
            "top_features": [item.to_dict() for item in self.top_features],
            "note": _note_text(),
        }


def _metadata_path_for_bundle(bundle_path: Path) -> Path:
    return bundle_path.with_suffix(".json")


def _local_vector_bundle_root() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "sae_vector_analysis_bundle" / "unsupervised_layer17_width_16k_l0_medium"


def resolve_default_vector_sae_analysis_bundle_paths() -> tuple[Path, Path]:
    local_root = _local_vector_bundle_root()
    for version in ("v2", "v1"):
        local_tensor = local_root / f"unsupervised_sae_vector_analysis_bundle_{version}.safetensors"
        local_metadata = local_root / f"unsupervised_sae_vector_analysis_bundle_{version}.json"
        if local_tensor.exists() and local_metadata.exists():
            return local_tensor, local_metadata
    manifest = load_manifest()
    if "unsupervised_sae_vector_analysis_bundle" not in manifest or "unsupervised_sae_vector_analysis_metadata" not in manifest:
        raise FileNotFoundError(
            "Could not find a local vector SAE analysis bundle, and the package manifest does not yet point to a remote one."
        )
    tensor_path = _download_hf_file(dict(manifest["unsupervised_sae_vector_analysis_bundle"]))
    metadata_path = _download_hf_file(dict(manifest["unsupervised_sae_vector_analysis_metadata"]))
    return tensor_path, metadata_path


def load_vector_sae_analysis_bundle(
    bundle_path: str | Path | None = None,
    *,
    metadata_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> VectorSAEAnalysisBundle:
    if bundle_path is None:
        resolved_tensor_path, resolved_metadata_path = resolve_default_vector_sae_analysis_bundle_paths()
        path = resolved_tensor_path
        metadata_path_obj = resolved_metadata_path if metadata_path is None else Path(metadata_path)
    else:
        path = Path(bundle_path)
        metadata_path_obj = _metadata_path_for_bundle(path) if metadata_path is None else Path(metadata_path)
    metadata = json.loads(metadata_path_obj.read_text(encoding="utf-8"))
    tensors = load_file(str(path), device=str(torch.device(device)))
    return VectorSAEAnalysisBundle(
        tensor_path=path,
        metadata_path=metadata_path_obj,
        metadata=metadata,
        w_dec=tensors["W_dec"].to(device=device, dtype=compute_dtype),
        transform_mean=tensors["transform_mean"].to(device=device, dtype=compute_dtype),
        transform_scale=tensors["transform_scale"].to(device=device, dtype=compute_dtype),
        direction=tensors["direction"].to(device=device, dtype=compute_dtype),
        midpoint_z=tensors["midpoint_z"].to(device=device, dtype=compute_dtype),
        intercept=float(tensors["intercept"].item()),
    )


class UnsupervisedSAEExplainer:
    def __init__(
        self,
        detector: FastAIDetector,
        bundle: VectorSAEAnalysisBundle,
    ):
        if detector.mode != "contrast":
            raise ValueError("SAE analysis is only supported for contrast mode.")
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

    def detector_scores_from_raw(self, raw_reps: torch.Tensor) -> torch.Tensor:
        z = (raw_reps - self.bundle.transform_mean.to(raw_reps.dtype)) / self.bundle.transform_scale.to(raw_reps.dtype)
        return z @ self.bundle.direction.to(raw_reps.dtype) + self.bundle.intercept

    def feature_geometry_from_raw(self, raw_reps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_dec = self.bundle.w_dec.to(raw_reps.dtype)
        z = (raw_reps - self.bundle.transform_mean.to(raw_reps.dtype)) / self.bundle.transform_scale.to(raw_reps.dtype)
        centered = z - self.bundle.midpoint_z.to(raw_reps.dtype)
        state_vs_midpoint = centered @ w_dec.transpose(0, 1)
        ai_association = w_dec @ self.bundle.direction.to(raw_reps.dtype)
        ai_net_push = state_vs_midpoint * ai_association.unsqueeze(0)
        return state_vs_midpoint, ai_association, ai_net_push

    def feature_lookup(self) -> dict[int, FeatureLookupRow]:
        if self._feature_lookup is None:
            self._feature_lookup = load_feature_lookup()
        return self._feature_lookup

    def _build_feature_items(
        self,
        *,
        state_vs_midpoint: torch.Tensor,
        ai_association: torch.Tensor,
        ai_net_push: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
    ) -> list[SAEFeatureContribution]:
        lookup = self.feature_lookup()
        items: list[SAEFeatureContribution] = []
        for feature_index in indices.tolist():
            state_value = float(state_vs_midpoint[feature_index].item())
            align_value = float(ai_association[feature_index].item())
            push_value = float(ai_net_push[feature_index].item())
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
                    state_vs_midpoint=state_value,
                    usual_association="ai" if align_value > 0 else "human",
                    ai_net_push=push_value,
                )
            )
            if len(items) >= top_k:
                break
        return items

    @staticmethod
    def _sorted_state_indices(row: torch.Tensor) -> torch.Tensor:
        candidate_indices = torch.nonzero(row > 0, as_tuple=False).squeeze(-1)
        if candidate_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=row.device)
        candidate_values = row[candidate_indices]
        order = torch.argsort(candidate_values, descending=True)
        return candidate_indices[order]

    def explain_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        top_k: int = 10,
    ) -> list[SAEExplanation]:
        raw_reps = self.predict_raw_representations(texts, batch_size=batch_size)
        detector_scores = self.detector_scores_from_raw(raw_reps)
        state_vs_midpoint, ai_association, ai_net_push = self.feature_geometry_from_raw(raw_reps)
        explanations: list[SAEExplanation] = []
        for index in range(len(texts)):
            row = state_vs_midpoint[index]
            sorted_indices = self._sorted_state_indices(row)
            top_features = self._build_feature_items(
                state_vs_midpoint=row,
                ai_association=ai_association,
                ai_net_push=ai_net_push[index],
                indices=sorted_indices,
                top_k=top_k,
            )
            explanations.append(
                SAEExplanation(
                    detector_score=float(detector_scores[index].item()),
                    total_ranked_features=int(row.numel()),
                    top_features=top_features,
                )
            )
        return explanations
