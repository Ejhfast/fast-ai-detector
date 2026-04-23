from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .inference import FastAIDetector


DEFAULT_LABEL_COLUMN = "fast_ai_detector_label"
DEFAULT_SCORE_COLUMN = "fast_ai_detector_score"
COMMON_TEXT_COLUMNS = ("text", "generation", "content", "body", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast local AI-text detector.")
    parser.add_argument("--mode", choices=["unsupervised", "raid-finetune"], default="unsupervised")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--output", default=None, help="Optional output .csv or .tsv path for file mode.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", default=None, help="Direct text to score.")
    source.add_argument("--input", default=None, help="Input .csv or .tsv file to score.")
    return parser.parse_args()


def detect_sep(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    if suffix == ".tsv":
        return "\t"
    raise ValueError(f"Unsupported file type for {path}. Expected .csv or .tsv.")


def resolve_text_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Missing text column {requested!r}. Available columns: {list(frame.columns)!r}")
        return requested
    for candidate in COMMON_TEXT_COLUMNS:
        if candidate in frame.columns:
            return candidate
    if len(frame.columns) == 1:
        return str(frame.columns[0])
    raise ValueError(
        "Could not infer the text column. Pass --text-column explicitly. "
        f"Available columns: {list(frame.columns)!r}"
    )


def score_direct_text(detector: FastAIDetector, text: str, batch_size: int) -> int:
    result = detector.score_texts([text], batch_size=batch_size)
    payload = {
        "mode": detector.mode,
        "label": result.labels[0],
        "score": result.scores[0],
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0


def score_file(
    detector: FastAIDetector,
    input_path: Path,
    output_path: Path | None,
    text_column: str | None,
    label_column: str,
    score_column: str,
    batch_size: int,
) -> int:
    sep = detect_sep(input_path)
    frame = pd.read_csv(input_path, sep=sep, low_memory=False)
    text_col = resolve_text_column(frame, text_column)
    predictions = detector.score_texts(frame[text_col].fillna("").astype(str).tolist(), batch_size=batch_size)
    scored = frame.copy()
    scored[label_column] = predictions.labels
    scored[score_column] = predictions.scores

    if output_path is None:
        scored.to_csv(sys.stdout, sep=sep, index=False)
        return 0

    out_sep = detect_sep(output_path)
    scored.to_csv(output_path, sep=out_sep, index=False)
    print(str(output_path), file=sys.stderr)
    return 0


def main() -> int:
    args = parse_args()
    detector = FastAIDetector(mode=args.mode, device=args.device)
    if args.text is not None:
        return score_direct_text(detector, args.text, batch_size=args.batch_size)
    return score_file(
        detector,
        input_path=Path(args.input),
        output_path=None if args.output is None else Path(args.output),
        text_column=args.text_column,
        label_column=args.label_column,
        score_column=args.score_column,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())

