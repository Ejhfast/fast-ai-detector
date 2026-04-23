from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from torch import nn
from transformers import AutoTokenizer

Mode = Literal["unsupervised", "raid-finetune"]


def _assets_root() -> Path:
    return Path(str(resources.files("fast_ai_detector").joinpath("assets")))


def load_manifest() -> dict[str, object]:
    return json.loads((_assets_root() / "manifest.json").read_text(encoding="utf-8"))


def _download_hf_bundle(bundle: dict[str, object]) -> Path:
    repo_id = str(bundle["repo_id"])
    filename = str(bundle["filename"])
    revision = bundle.get("revision")
    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=None if revision is None else str(revision),
    )
    if isinstance(cached, str):
        return Path(cached)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            revision=None if revision is None else str(revision),
        )
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device)


class MeanResidStudent(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_token_id: int,
        max_length: int,
        token_embed_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_dim: int,
        dropout: float,
        output_dim: int,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, token_embed_dim, padding_idx=pad_token_id)
        self.embed_proj = nn.Linear(token_embed_dim, d_model) if token_embed_dim != d_model else nn.Identity()
        self.pos_embed = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.embed_proj(self.token_embed(input_ids)) + self.pos_embed(positions)
        key_padding_mask = ~attention_mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.out(pooled)


class DensePredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(_assets_root() / "tokenizer", use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_batch(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return {
        "input_ids": encoded["input_ids"].to(device, non_blocking=True),
        "attention_mask": encoded["attention_mask"].to(device, non_blocking=True).bool(),
    }


@dataclass
class StudentState:
    model: MeanResidStudent
    tokenizer: object
    max_length: int
    device: torch.device
    target_mean: torch.Tensor
    target_scale: torch.Tensor


@dataclass
class UnsupervisedState:
    student: StudentState
    mean: np.ndarray
    scale: np.ndarray
    direction: np.ndarray
    intercept: float


@dataclass
class FinetuneState:
    student: StudentState
    head: DensePredictionHead
    head_input_space: str


def _build_student_from_checkpoint(checkpoint: dict[str, object], device: torch.device) -> StudentState:
    config = checkpoint["config"]
    tokenizer = load_tokenizer()
    target_mean = torch.tensor(checkpoint["target_mean"], dtype=torch.float32, device=device)
    target_scale = torch.tensor(checkpoint["target_scale"], dtype=torch.float32, device=device)
    model = MeanResidStudent(
        vocab_size=len(tokenizer),
        pad_token_id=int(tokenizer.pad_token_id),
        max_length=int(config["max_length"]),
        token_embed_dim=int(config["token_embed_dim"]),
        d_model=int(config["d_model"]),
        n_layers=int(config["n_layers"]),
        n_heads=int(config["n_heads"]),
        mlp_dim=int(float(config["d_model"]) * float(config["mlp_ratio"])),
        dropout=float(config["dropout"]),
        output_dim=int(target_mean.numel()),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return StudentState(
        model=model,
        tokenizer=tokenizer,
        max_length=int(config["max_length"]),
        device=device,
        target_mean=target_mean,
        target_scale=target_scale,
    )


def load_unsupervised_state(device: torch.device) -> UnsupervisedState:
    manifest = load_manifest()
    checkpoint = torch.load(
        _download_hf_bundle(dict(manifest["student_bundle"])),
        map_location="cpu",
        weights_only=False,
    )
    delta = np.load(_assets_root() / "models" / str(manifest["unsupervised_delta"]))
    student = _build_student_from_checkpoint(checkpoint, device)
    return UnsupervisedState(
        student=student,
        mean=delta["mean"].astype(np.float32),
        scale=delta["scale"].astype(np.float32),
        direction=delta["direction"].astype(np.float32),
        intercept=float(delta["intercept"][0]),
    )


def load_finetune_state(device: torch.device) -> FinetuneState:
    manifest = load_manifest()
    checkpoint = torch.load(
        _download_hf_bundle(dict(manifest["finetune_bundle"])),
        map_location="cpu",
        weights_only=False,
    )
    student_checkpoint = {
        "config": checkpoint["student_config"],
        "target_mean": checkpoint["target_mean"],
        "target_scale": checkpoint["target_scale"],
        "model_state_dict": checkpoint["student_state_dict"],
    }
    student = _build_student_from_checkpoint(student_checkpoint, device)
    head_config = checkpoint["head_config"]
    head = DensePredictionHead(
        input_dim=int(student.target_mean.numel()),
        hidden_dim=int(head_config["head_hidden_dim"]),
        dropout=float(head_config["head_dropout"]),
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    return FinetuneState(
        student=student,
        head=head,
        head_input_space=str(head_config.get("head_input_space", "z")),
    )
