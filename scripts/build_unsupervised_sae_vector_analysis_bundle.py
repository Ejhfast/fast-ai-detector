from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
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
    parser.add_argument("--decoder-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "sae_vector_analysis_bundle" / "unsupervised_layer17_width_16k_l0_medium"),
    )
    parser.add_argument("--bundle-name", default="unsupervised_sae_vector_analysis_bundle_v2")
    parser.add_argument(
        "--stats-path",
        default=str(ROOT / "outputs" / "raid_full_student_delta_stats.npz"),
        help="Optional cached AI/human class means for the full RAID student cache. Recomputed if missing.",
    )
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


def compute_or_load_full_raid_student_stats(stats_path: Path, cache_dir: Path) -> dict[str, np.ndarray | int]:
    if stats_path.exists():
        with np.load(stats_path) as data:
            return {
                "mu_ai": data["mu_ai"].astype(np.float32),
                "mu_human": data["mu_human"].astype(np.float32),
                "n_ai": int(data["n_ai"][0]),
                "n_human": int(data["n_human"][0]),
            }

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    features = np.load(cache_dir / str(manifest["feature_file"]), mmap_mode="r")
    labels = pd.read_csv(cache_dir / "metadata.csv", usecols=["label_ai"])["label_ai"].to_numpy(dtype=np.int8)
    chunk_size = 100_000
    sum_ai = np.zeros(features.shape[2], dtype=np.float64)
    sum_human = np.zeros(features.shape[2], dtype=np.float64)
    n_ai = 0
    n_human = 0

    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        batch = np.asarray(features[start:end, 0, :], dtype=np.float32)
        is_ai = labels[start:end] == 1
        if is_ai.any():
            sum_ai += batch[is_ai].sum(axis=0, dtype=np.float64)
            n_ai += int(is_ai.sum())
        is_human = ~is_ai
        if is_human.any():
            sum_human += batch[is_human].sum(axis=0, dtype=np.float64)
            n_human += int(is_human.sum())

    mu_ai = (sum_ai / max(1, n_ai)).astype(np.float32)
    mu_human = (sum_human / max(1, n_human)).astype(np.float32)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        mu_ai=mu_ai,
        mu_human=mu_human,
        n_ai=np.asarray([n_ai], dtype=np.int64),
        n_human=np.asarray([n_human], dtype=np.int64),
    )
    return {
        "mu_ai": mu_ai,
        "mu_human": mu_human,
        "n_ai": n_ai,
        "n_human": n_human,
    }


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
    cache_dir = Path(str(manifest["raid_cache_dir"]))
    stats_path = Path(args.stats_path)
    stats = compute_or_load_full_raid_student_stats(stats_path, cache_dir)

    transform_mean = delta["mean"].astype(np.float32)
    transform_scale = delta["scale"].astype(np.float32)
    direction = delta["direction"].astype(np.float32)
    intercept = float(delta["intercept"][0])
    mu_ai_z = (stats["mu_ai"] - transform_mean) / transform_scale
    mu_human_z = (stats["mu_human"] - transform_mean) / transform_scale
    midpoint_z = 0.5 * (mu_ai_z + mu_human_z)

    decoder_dtype = torch.float16 if args.decoder_dtype == "float16" else torch.float32
    bundle_tensors = {
        "W_dec": sae_tensors["w_dec"].to(dtype=decoder_dtype).contiguous(),
        "transform_mean": torch.from_numpy(transform_mean).float().contiguous(),
        "transform_scale": torch.from_numpy(transform_scale).float().contiguous(),
        "direction": torch.from_numpy(direction).float().contiguous(),
        "midpoint_z": torch.from_numpy(midpoint_z.astype(np.float32)).float().contiguous(),
        "intercept": torch.tensor([intercept], dtype=torch.float32),
    }
    save_file(bundle_tensors, str(bundle_path))

    metadata = {
        "bundle_format": 2,
        "bundle_kind": "fast_ai_detector_unsupervised_sae_vector_analysis",
        "tensor_file": bundle_path.name,
        "source_sae_release": args.sae_release,
        "source_sae_id": args.sae_id,
        "source_sae_dir": str(sae_dir.resolve()),
        "source_detector_mode": "unsupervised",
        "source_detector_manifest": manifest,
        "source_delta_path": str(delta_path.resolve()),
        "source_stats_path": str(stats_path.resolve()),
        "n_ai": int(stats["n_ai"]),
        "n_human": int(stats["n_human"]),
        "d_in": int(sae_tensors["w_dec"].shape[1]),
        "d_sae": int(sae_tensors["w_dec"].shape[0]),
        "decoder_storage_dtype": args.decoder_dtype,
        "tensor_dtypes": {name: str(tensor.dtype).replace("torch.", "") for name, tensor in bundle_tensors.items()},
        "transform_definition": "z = (raw_rep - transform_mean) / transform_scale",
        "midpoint_definition": "midpoint_z = 0.5 * (mu_ai_z + mu_human_z)",
        "example_score_vector_definition": "((z - midpoint_z) * direction)",
        "feature_score_definition": "W_dec @ (((z - midpoint_z) * direction))",
        "detector_score_definition": "z @ direction + intercept",
        "notes": [
            "This bundle is for vector-space SAE interpretation of the midpoint-centered per-example score vector.",
            "Positive feature scores push toward AI; negative feature scores push toward human.",
            "No SAE encoder or thresholding is used in this analysis path.",
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
