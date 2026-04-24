from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from fast_ai_detector.inference import FastAIDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a RAID-format CSV with fast-ai-detector and write RAID leaderboard "
            "predictions.json records of the form [{'id': ..., 'score': ...}, ...]."
        )
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mode", choices=["unsupervised", "raid-finetune"], default="raid-finetune")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--csv-chunksize", type=int, default=4096)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="generation")
    return parser.parse_args()


def count_rows(csv_path: Path) -> int:
    total = 0
    for chunk in pd.read_csv(csv_path, usecols=[0], chunksize=100000):
        total += len(chunk)
    return total


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    detector = FastAIDetector(mode=args.mode, device=args.device)
    total_rows = count_rows(input_csv)

    with output_json.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        wrote_any = False
        progress = tqdm(total=total_rows, desc=f"Scoring RAID {args.mode} CSV")
        for chunk in pd.read_csv(
            input_csv,
            usecols=[args.id_col, args.text_col],
            chunksize=args.csv_chunksize,
            dtype="string",
            keep_default_na=False,
        ):
            texts = chunk[args.text_col].astype(str).tolist()
            ids = chunk[args.id_col].astype(str).tolist()
            scores = detector.score_texts(texts, batch_size=args.batch_size).scores
            for row_id, score in zip(ids, scores, strict=False):
                if wrote_any:
                    handle.write(",\n")
                json.dump({"id": row_id, "score": float(score)}, handle, ensure_ascii=True)
                wrote_any = True
            progress.update(len(chunk))
        progress.close()
        handle.write("\n]\n")

    summary = {
        "input_csv": str(input_csv.resolve()),
        "output_json": str(output_json.resolve()),
        "mode": args.mode,
        "device": args.device,
        "n_rows": total_rows,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
