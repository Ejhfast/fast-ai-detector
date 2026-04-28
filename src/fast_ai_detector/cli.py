from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .inference import FastAIDetector
from .models import UnsupervisedState
from .sae_analysis import UnsupervisedSAEExplainer, load_vector_sae_analysis_bundle


DEFAULT_LABEL_COLUMN = "fast_ai_detector_label"
DEFAULT_SCORE_COLUMN = "fast_ai_detector_score"
DEFAULT_SCALE_COLUMN = "fast_ai_detector_human_ai_scale"
COMMON_TEXT_COLUMNS = ("text", "generation", "content", "body", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast local AI-text detector.")
    parser.add_argument("--mode", choices=["contrast", "raid-finetune"], default="contrast")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--long-doc-mode",
        choices=["truncate", "chunk-mean"],
        default="truncate",
        help="How to handle inputs longer than the model context window.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON for --text mode instead of human-readable tables.")
    parser.add_argument("--explain-sae", action="store_true", help="Include vector-space SAE feature analysis (contrast mode only).")
    parser.add_argument("--sae-top-k", type=int, default=10)
    parser.add_argument("--sae-output-jsonl", default=None, help="Optional JSONL sidecar path for SAE explanations in file mode.")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--human-ai-scale-column", default=DEFAULT_SCALE_COLUMN)
    parser.add_argument("--output", default=None, help="Optional output .csv or .tsv path for file mode.")
    parser.add_argument(
        "--export-vectors",
        default=None,
        help=(
            "Write document-level student vectors instead of detector scores. "
            "Supported suffixes: .npy, .npz, .csv, .tsv."
        ),
    )
    parser.add_argument(
        "--vector-space",
        choices=["raw", "student-z", "detector-z"],
        default="raw",
        help=(
            "Representation space for --export-vectors. raw is the predicted Gemma layer-17 mean residual; "
            "student-z applies the student target normalization; detector-z applies the contrast detector normalization."
        ),
    )
    parser.add_argument(
        "--vector-metadata-output",
        default=None,
        help="Optional .csv/.tsv metadata path for --export-vectors. Defaults to <vectors>.metadata.csv for .npy output.",
    )
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


def build_sae_explainer(detector: FastAIDetector) -> UnsupervisedSAEExplainer:
    if detector.mode != "contrast":
        raise ValueError("SAE analysis is only available in contrast mode.")
    bundle = load_vector_sae_analysis_bundle(device=detector.device)
    return UnsupervisedSAEExplainer(detector, bundle)


def _print_aligned_table(headers: list[str], rows: list[list[object]]) -> None:
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in string_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print(format_row(headers))
    for row in string_rows:
        print(format_row(row))


def _format_score_value(value: float) -> str:
    return f"{value:.6f}"


def _vector_columns(width: int) -> list[str]:
    return [f"vector_{index:04d}" for index in range(width)]


def _metadata_path_for_vectors(vector_output_path: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        return requested
    if vector_output_path.suffix.lower() == ".npy":
        return vector_output_path.with_suffix(".metadata.csv")
    return None


def _representations_in_space(detector: FastAIDetector, texts: list[str], batch_size: int, vector_space: str) -> np.ndarray:
    raw_reps = detector.predict_raw_representations(texts, batch_size=batch_size).float()
    if vector_space == "raw":
        reps = raw_reps
    elif vector_space == "student-z":
        student = detector.state.student
        reps = (raw_reps - student.target_mean.to(raw_reps.dtype)) / student.target_scale.to(raw_reps.dtype)
    elif vector_space == "detector-z":
        if not isinstance(detector.state, UnsupervisedState):
            raise ValueError("--vector-space detector-z is only available with --mode contrast.")
        mean = raw_reps.new_tensor(detector.state.mean)
        scale = raw_reps.new_tensor(detector.state.scale)
        reps = (raw_reps - mean) / scale
    else:
        raise ValueError(f"Unsupported vector space: {vector_space!r}")
    return reps.cpu().numpy().astype(np.float32, copy=False)


def _vector_manifest(
    *,
    detector: FastAIDetector,
    vector_space: str,
    text_column: str | None,
    num_examples: int,
    vector_dim: int,
) -> dict[str, object]:
    student = detector.state.student
    return {
        "mode": detector.mode,
        "long_doc_mode": detector.long_doc_mode,
        "vector_space": vector_space,
        "text_column": text_column,
        "num_examples": int(num_examples),
        "vector_dim": int(vector_dim),
        "dtype": "float32",
        "student_max_length": int(student.max_length),
        "definition": {
            "raw": "Predicted Gemma 3 4B layer-17 mean residual representation from the distilled student.",
            "student-z": "raw normalized by the student checkpoint target_mean and target_scale.",
            "detector-z": "raw normalized by the contrast detector mean and scale.",
        }[vector_space],
    }


def _write_vector_outputs(
    *,
    vectors: np.ndarray,
    vector_output_path: Path,
    metadata: pd.DataFrame | None,
    metadata_output_path: Path | None,
    manifest: dict[str, object],
) -> None:
    suffix = vector_output_path.suffix.lower()
    vector_output_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".npy":
        np.save(vector_output_path, vectors)
    elif suffix == ".npz":
        payload: dict[str, object] = {"vectors": vectors, "manifest_json": json.dumps(manifest, sort_keys=True)}
        if metadata is not None:
            payload["row_index"] = metadata["row_index"].to_numpy(dtype=np.int64)
        np.savez_compressed(vector_output_path, **payload)
    elif suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        vector_frame = pd.DataFrame(vectors, columns=_vector_columns(vectors.shape[1]))
        output_frame = vector_frame if metadata is None else pd.concat([metadata.reset_index(drop=True), vector_frame], axis=1)
        output_frame.to_csv(vector_output_path, sep=sep, index=False)
    else:
        raise ValueError(f"Unsupported vector output suffix {suffix!r}. Use .npy, .npz, .csv, or .tsv.")

    manifest_path = vector_output_path.with_suffix(vector_output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if metadata is not None and metadata_output_path is not None:
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
        sep = detect_sep(metadata_output_path)
        metadata.to_csv(metadata_output_path, sep=sep, index=False)

    print(str(vector_output_path), file=sys.stderr)


def export_vectors_direct_text(
    detector: FastAIDetector,
    text: str,
    batch_size: int,
    *,
    vector_output_path: Path,
    vector_space: str,
    metadata_output_path: Path | None,
) -> int:
    vectors = _representations_in_space(detector, [text], batch_size, vector_space)
    metadata = pd.DataFrame({"row_index": [0], "text": [text]})
    manifest = _vector_manifest(
        detector=detector,
        vector_space=vector_space,
        text_column="text",
        num_examples=vectors.shape[0],
        vector_dim=vectors.shape[1],
    )
    _write_vector_outputs(
        vectors=vectors,
        vector_output_path=vector_output_path,
        metadata=metadata,
        metadata_output_path=_metadata_path_for_vectors(vector_output_path, metadata_output_path),
        manifest=manifest,
    )
    return 0


def export_vectors_file(
    detector: FastAIDetector,
    input_path: Path,
    text_column: str | None,
    batch_size: int,
    *,
    vector_output_path: Path,
    vector_space: str,
    metadata_output_path: Path | None,
) -> int:
    sep = detect_sep(input_path)
    frame = pd.read_csv(input_path, sep=sep, low_memory=False)
    text_col = resolve_text_column(frame, text_column)
    texts = frame[text_col].fillna("").astype(str).tolist()
    vectors = _representations_in_space(detector, texts, batch_size, vector_space)
    metadata = frame.copy()
    metadata.insert(0, "row_index", np.arange(len(metadata), dtype=np.int64))
    manifest = _vector_manifest(
        detector=detector,
        vector_space=vector_space,
        text_column=text_col,
        num_examples=vectors.shape[0],
        vector_dim=vectors.shape[1],
    )
    _write_vector_outputs(
        vectors=vectors,
        vector_output_path=vector_output_path,
        metadata=metadata,
        metadata_output_path=_metadata_path_for_vectors(vector_output_path, metadata_output_path),
        manifest=manifest,
    )
    return 0


def _print_direct_text_report(
    *,
    detector: FastAIDetector,
    label: str,
    score: float,
    human_ai_scale: float,
    explanation: dict[str, object] | None,
) -> None:
    _print_aligned_table(
        ["mode", "label", "score", "human_ai_scale"],
        [[detector.mode, label, _format_score_value(score), _format_score_value(human_ai_scale)]],
    )
    if explanation is None:
        return
    print()
    feature_rows: list[list[object]] = []
    for item in explanation.get("top_features", []):
        if not isinstance(item, dict):
            continue
        feature_rows.append(
            [
                item.get("feature_index", ""),
                item.get("title", ""),
                _format_score_value(float(item.get("state_vs_midpoint", 0.0))),
                item.get("usual_association", ""),
                _format_score_value(float(item.get("ai_net_push", 0.0))),
            ]
        )
    if not feature_rows:
        print("No explained SAE features found.")
        return
    _print_aligned_table(
        ["feature_index", "title", "state_vs_midpoint", "usual_assoc", "ai_net_push"],
        feature_rows,
    )


def score_direct_text(
    detector: FastAIDetector,
    text: str,
    batch_size: int,
    *,
    emit_json: bool,
    explain_sae: bool,
    sae_top_k: int,
) -> int:
    result = detector.score_texts([text], batch_size=batch_size)
    payload = {
        "mode": detector.mode,
        "label": result.labels[0],
        "score": result.scores[0],
        "human_ai_scale": result.pcts[0],
    }
    explanation_dict: dict[str, object] | None = None
    if explain_sae:
        explanation = build_sae_explainer(detector).explain_texts([text], batch_size=batch_size, top_k=sae_top_k)[0]
        explanation_dict = explanation.to_dict()
        payload["sae_analysis"] = explanation_dict
    if emit_json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        _print_direct_text_report(
            detector=detector,
            label=result.labels[0],
            score=result.scores[0],
            human_ai_scale=result.pcts[0],
            explanation=explanation_dict,
        )
    return 0


def score_file(
    detector: FastAIDetector,
    input_path: Path,
    output_path: Path | None,
    text_column: str | None,
    label_column: str,
    score_column: str,
    human_ai_scale_column: str,
    batch_size: int,
    *,
    explain_sae: bool,
    sae_top_k: int,
    sae_output_jsonl: Path | None,
) -> int:
    sep = detect_sep(input_path)
    frame = pd.read_csv(input_path, sep=sep, low_memory=False)
    text_col = resolve_text_column(frame, text_column)
    predictions = detector.score_texts(frame[text_col].fillna("").astype(str).tolist(), batch_size=batch_size)
    scored = frame.copy()
    scored[label_column] = predictions.labels
    scored[score_column] = predictions.scores
    scored[human_ai_scale_column] = predictions.pcts

    if explain_sae:
        if sae_output_jsonl is None:
            raise ValueError("Pass --sae-output-jsonl when using --explain-sae with file input.")
        explanations = build_sae_explainer(detector).explain_texts(
            frame[text_col].fillna("").astype(str).tolist(),
            batch_size=batch_size,
            top_k=sae_top_k,
        )
        sae_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with sae_output_jsonl.open("w", encoding="utf-8") as handle:
            for row_index, (prediction_label, prediction_score, prediction_scale, explanation) in enumerate(
                zip(predictions.labels, predictions.scores, predictions.pcts, explanations, strict=False)
            ):
                payload = {
                    "row_index": row_index,
                    "label": prediction_label,
                    "score": prediction_score,
                    "human_ai_scale": prediction_scale,
                    "sae_analysis": explanation.to_dict(),
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    if output_path is None:
        scored.to_csv(sys.stdout, sep=sep, index=False)
        return 0

    out_sep = detect_sep(output_path)
    scored.to_csv(output_path, sep=out_sep, index=False)
    print(str(output_path), file=sys.stderr)
    return 0


def main() -> int:
    args = parse_args()
    detector = FastAIDetector(mode=args.mode, device=args.device, long_doc_mode=args.long_doc_mode)
    if args.export_vectors is not None:
        vector_output_path = Path(args.export_vectors)
        metadata_output_path = None if args.vector_metadata_output is None else Path(args.vector_metadata_output)
        if args.text is not None:
            return export_vectors_direct_text(
                detector,
                args.text,
                batch_size=args.batch_size,
                vector_output_path=vector_output_path,
                vector_space=args.vector_space,
                metadata_output_path=metadata_output_path,
            )
        return export_vectors_file(
            detector,
            input_path=Path(args.input),
            text_column=args.text_column,
            batch_size=args.batch_size,
            vector_output_path=vector_output_path,
            vector_space=args.vector_space,
            metadata_output_path=metadata_output_path,
        )
    if args.text is not None:
        return score_direct_text(
            detector,
            args.text,
            batch_size=args.batch_size,
            emit_json=bool(args.json),
            explain_sae=bool(args.explain_sae),
            sae_top_k=int(args.sae_top_k),
        )
    return score_file(
        detector,
        input_path=Path(args.input),
        output_path=None if args.output is None else Path(args.output),
        text_column=args.text_column,
        label_column=args.label_column,
        score_column=args.score_column,
        human_ai_scale_column=args.human_ai_scale_column,
        batch_size=args.batch_size,
        explain_sae=bool(args.explain_sae),
        sae_top_k=int(args.sae_top_k),
        sae_output_jsonl=None if args.sae_output_jsonl is None else Path(args.sae_output_jsonl),
    )


if __name__ == "__main__":
    raise SystemExit(main())
