# fast-ai-detector

Local CLI for fast AI-text scoring with a bundled tokenizer and remotely hosted model bundles.

Initial modes:

- `unsupervised` (default): RAID-trained student-space z-score delta
- `raid-finetune`: epoch-7 RAID finetuned classifier

## Install

```bash
pip install -e .
```

The tokenizer ships with the package. On first use, the selected model bundle is downloaded from Hugging Face and cached locally.

## Score Direct Text

```bash
fast-ai-detector --text "This is a test."
fast-ai-detector --mode raid-finetune --device cuda --text "This is a test."
```

The direct-text mode prints a JSON object with the predicted label and score.

It also includes:

- `human_ai_scale`: RAID-reference scale on a `0 -> human` to `100 -> ai` axis, anchored so `50` is the score boundary

## Score CSV / TSV

Input files must be `.csv` or `.tsv`. The tool preserves existing column order and appends:

- `fast_ai_detector_label`
- `fast_ai_detector_score`
- `fast_ai_detector_human_ai_scale`

Examples:

```bash
fast-ai-detector --input rows.csv --text-column text --output scored.csv
fast-ai-detector --input rows.tsv --text-column generation > scored.tsv
```

## Notes

- `--device auto` uses CUDA if available, otherwise CPU.
- Positive scores indicate stronger evidence for AI in both modes.
- `human_ai_scale` is not a probability. It is a signed reference scale on the detector score axis:
  `0 = strongly human side`, `50 = near boundary`, `100 = strongly AI side`.
- `unsupervised` downloads `unignorant/fast-ai-detector`.
- `raid-finetune` downloads `unignorant/fast-ai-detector-raid-finetune`.
