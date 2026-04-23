from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .models import (
    FinetuneState,
    Mode,
    UnsupervisedState,
    load_finetune_state,
    load_unsupervised_state,
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
    def __init__(self, mode: Mode = "unsupervised", device: str = "auto", load_percentiles: bool = True):
        self.mode = mode
        self.device = resolve_device(device)
        self.percentile_artifact = load_percentile_artifact_for_mode(mode) if load_percentiles else None
        if mode == "unsupervised":
            self.state = load_unsupervised_state(self.device)
        elif mode == "raid-finetune":
            self.state = load_finetune_state(self.device)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _score_unsupervised(self, texts: list[str], batch_size: int) -> list[float]:
        state: UnsupervisedState = self.state
        student = state.student
        scores: list[np.ndarray] = []
        use_amp = student.device.type == "cuda"
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                tokens = tokenize_batch(student.tokenizer, batch_texts, student.max_length, student.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred_z = student.model(tokens["input_ids"], tokens["attention_mask"])
                    pred_raw = pred_z * student.target_scale + student.target_mean
                pred_raw_np = pred_raw.float().cpu().numpy().astype(np.float32)
                z = (pred_raw_np - state.mean) / state.scale
                batch_scores = z @ state.direction + state.intercept
                scores.append(batch_scores.astype(np.float64))
        return np.concatenate(scores).tolist() if scores else []

    def _score_finetune(self, texts: list[str], batch_size: int) -> list[float]:
        state: FinetuneState = self.state
        student = state.student
        scores: list[np.ndarray] = []
        use_amp = student.device.type == "cuda"
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                tokens = tokenize_batch(student.tokenizer, batch_texts, student.max_length, student.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred_z = student.model(tokens["input_ids"], tokens["attention_mask"])
                    if state.head_input_space == "z":
                        rep = pred_z
                    else:
                        rep = pred_z * student.target_scale + student.target_mean
                    logits = state.head(rep)
                scores.append(logits.float().cpu().numpy().astype(np.float64))
        return np.concatenate(scores).tolist() if scores else []

    def score_texts(self, texts: Sequence[str], batch_size: int = 64) -> PredictionBatch:
        clean_texts = ["" if text is None else str(text) for text in texts]
        if self.mode == "unsupervised":
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
