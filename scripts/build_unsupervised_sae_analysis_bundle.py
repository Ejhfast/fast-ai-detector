from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fast_ai_detector.models import load_manifest  # noqa: E402


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
    parser.add_argument("--sae-release", default="gemma-scope-2-4b-pt-res-all")
    parser.add_argument("--sae-id", default="layer_17_width_16k_l0_medium")
    parser.add_argument("--encoder-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "sae_analysis_bundle" / "unsupervised_layer17_width_16k_l0_medium"),
    )
    parser.add_argument("--bundle-name", default="unsupervised_sae_analysis_bundle_v1")
    return parser.parse_args()


def assets_root() -> Path:
    return Path(str(resources.files("fast_ai_detector").joinpath("assets")))


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


def build_bundle(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / f"{args.bundle_name}.safetensors"
    metadata_path = output_dir / f"{args.bundle_name}.json"

    sae_dir = resolve_local_sae_dir(args.sae_release, args.sae_id)
    sae_tensors = load_file(str(sae_dir / "params.safetensors"), device="cpu")
    manifest = load_manifest()
    delta_path = assets_root() / "models" / str(manifest["unsupervised_delta"])
    delta = np.load(delta_path)

    mean = delta["mean"].astype(np.float32)
    scale = delta["scale"].astype(np.float32)
    direction = delta["direction"].astype(np.float32)
    intercept = float(delta["intercept"][0])
    raw_readout = direction / scale
    raw_intercept = intercept - float(np.dot(mean / scale, direction))

    w_enc = sae_tensors["w_enc"].contiguous()
    w_dec = sae_tensors["w_dec"].float().contiguous()
    b_enc = sae_tensors["b_enc"].float().contiguous()
    b_dec = sae_tensors["b_dec"].float().contiguous()
    threshold = sae_tensors["threshold"].float().contiguous()
    raw_readout_t = torch.from_numpy(raw_readout).float()
    feature_alignment = (w_dec @ raw_readout_t).contiguous()
    sae_score_offset = raw_intercept + float(torch.dot(b_dec, raw_readout_t).item())

    encoder_dtype = torch.float16 if args.encoder_dtype == "float16" else torch.float32
    bundle_tensors = {
        "W_enc": w_enc.to(dtype=encoder_dtype),
        "b_enc": b_enc,
        "threshold": threshold,
        "raw_readout": raw_readout_t,
        "feature_alignment": feature_alignment,
        "raw_intercept": torch.tensor([raw_intercept], dtype=torch.float32),
        "sae_score_offset": torch.tensor([sae_score_offset], dtype=torch.float32),
    }
    save_file(bundle_tensors, str(bundle_path))

    metadata = {
        "bundle_format": 1,
        "bundle_kind": "fast_ai_detector_unsupervised_sae_analysis",
        "tensor_file": bundle_path.name,
        "source_sae_release": args.sae_release,
        "source_sae_id": args.sae_id,
        "source_sae_dir": str(sae_dir.resolve()),
        "source_detector_mode": "unsupervised",
        "source_detector_manifest": manifest,
        "source_delta_path": str(delta_path.resolve()),
        "architecture": "jumprelu",
        "d_in": int(w_enc.shape[0]),
        "d_sae": int(w_enc.shape[1]),
        "encoder_storage_dtype": args.encoder_dtype,
        "tensor_dtypes": {name: str(tensor.dtype).replace("torch.", "") for name, tensor in bundle_tensors.items()},
        "sae_score_definition": "sum(feature_acts * feature_alignment) + sae_score_offset",
        "detector_score_definition": "raw_rep @ raw_readout + raw_intercept",
        "notes": [
            "Feature activations are computed as hidden_pre = raw_rep @ W_enc + b_enc; feature = hidden_pre if hidden_pre > threshold else 0.",
            "Positive feature contributions push toward AI; negative contributions push toward human.",
            "sae_score is an SAE-reconstruction approximation of the detector score, not the exact detector score.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return bundle_path, metadata_path


def main() -> None:
    args = parse_args()
    bundle_path, metadata_path = build_bundle(args)
    print(json.dumps({"bundle_path": str(bundle_path), "metadata_path": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
