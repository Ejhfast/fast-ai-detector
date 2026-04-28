from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .models import (
    FinetuneState,
    LongDocMode,
    Mode,
    UnsupervisedState,
    build_chunk_records,
    load_finetune_state,
    load_unsupervised_state,
    pack_chunk_records,
    resolve_device,
    tokenize_batch,
)
from .percentiles import load_percentile_artifact_for_mode, score_percentiles


@dataclass
class PredictionBatch:
    labels: list[str]
    scores: list[float]
    pcts: list[float]


class FastAIDetector:
    def __init__(
        self,
        mode: Mode = "contrast",
        device: str = "auto",
        load_percentiles: bool = True,
        long_doc_mode: LongDocMode = "truncate",
    ):
        self.mode = mode
        self.device = resolve_device(device)
        self.long_doc_mode = long_doc_mode
        self.percentile_artifact = (
            load_percentile_artifact_for_mode(self.mode) if load_percentiles and self.long_doc_mode == "truncate" else None
        )
        if self.mode == "contrast":
            self.state = load_unsupervised_state(self.device)
        elif self.mode == "raid-finetune":
            self.state = load_finetune_state(self.device)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _predict_truncated_raw_representations(self, texts: list[str], batch_size: int) -> torch.Tensor:
        student = self.state.student
        reps: list[torch.Tensor] = []
        use_amp = student.device.type == "cuda"
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                tokens = tokenize_batch(student.tokenizer, batch_texts, student.max_length, student.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred_z = student.model(tokens["input_ids"], tokens["attention_mask"])
                    pred_raw = pred_z * student.target_scale + student.target_mean
                reps.append(pred_raw.float())
        if not reps:
            return torch.empty((0, int(student.target_mean.numel())), device=student.device, dtype=torch.float32)
        return torch.cat(reps, dim=0)

    def _predict_chunk_mean_raw_representations(self, texts: list[str], batch_size: int) -> torch.Tensor:
        student = self.state.student
        d_out = int(student.target_mean.numel())
        if not texts:
            return torch.empty((0, d_out), device=student.device, dtype=torch.float32)
        use_amp = student.device.type == "cuda"
        doc_weighted_sum = torch.zeros((len(texts), d_out), device=student.device, dtype=torch.float32)
        doc_weight_total = torch.zeros(len(texts), device=student.device, dtype=torch.float32)
        chunk_records = build_chunk_records(student.tokenizer, texts, student.max_length)
        with torch.no_grad():
            for start in range(0, len(chunk_records), batch_size):
                chunk_batch = pack_chunk_records(
                    student.tokenizer,
                    chunk_records[start : start + batch_size],
                    student.max_length,
                    student.device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred_z = student.model(chunk_batch["input_ids"], chunk_batch["attention_mask"])
                    pred_raw = pred_z * student.target_scale + student.target_mean
                pred_raw = pred_raw.float()
                weights = chunk_batch["chunk_weights"].to(dtype=pred_raw.dtype)
                weighted_pred_raw = pred_raw * weights.unsqueeze(1)
                doc_weighted_sum.index_add_(0, chunk_batch["doc_indices"], weighted_pred_raw)
                doc_weight_total.index_add_(0, chunk_batch["doc_indices"], weights)
        return doc_weighted_sum / doc_weight_total.clamp_min(1.0).unsqueeze(1)

    def predict_raw_representations(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        clean_texts = ["" if text is None else str(text) for text in texts]
        if self.long_doc_mode == "truncate":
            return self._predict_truncated_raw_representations(clean_texts, batch_size=batch_size)
        if self.long_doc_mode == "chunk-mean":
            return self._predict_chunk_mean_raw_representations(clean_texts, batch_size=batch_size)
        raise ValueError(f"Unsupported long_doc_mode: {self.long_doc_mode}")

    def _score_unsupervised(self, texts: list[str], batch_size: int) -> list[float]:
        state: UnsupervisedState = self.state
        pred_raw_np = self.predict_raw_representations(texts, batch_size=batch_size).float().cpu().numpy().astype(np.float32)
        z = (pred_raw_np - state.mean) / state.scale
        scores = z @ state.direction + state.intercept
        return scores.astype(np.float64).tolist()

    def _score_finetune(self, texts: list[str], batch_size: int) -> list[float]:
        state: FinetuneState = self.state
        student = state.student
        raw_reps = self.predict_raw_representations(texts, batch_size=batch_size)
        with torch.no_grad():
            if state.head_input_space == "z":
                rep = (raw_reps - student.target_mean) / student.target_scale
            else:
                rep = raw_reps
            logits = state.head(rep)
        return logits.float().cpu().numpy().astype(np.float64).tolist()

    def score_texts(self, texts: Sequence[str], batch_size: int = 64) -> PredictionBatch:
        clean_texts = ["" if text is None else str(text) for text in texts]
        if self.mode == "contrast":
            scores = self._score_unsupervised(clean_texts, batch_size=batch_size)
        else:
            scores = self._score_finetune(clean_texts, batch_size=batch_size)
        if self.percentile_artifact is None:
            pcts_np = np.full(len(scores), np.nan, dtype=np.float64)
        else:
            pcts_np = score_percentiles(np.asarray(scores, dtype=np.float64), self.percentile_artifact)
        labels = ["ai" if score >= 0 else "human" for score in scores]
        return PredictionBatch(
            labels=labels,
            scores=[float(score) for score in scores],
            pcts=[float(value) for value in pcts_np],
        )
