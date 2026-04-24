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

The direct-text mode prints a small human-readable TSV report by default.

It also includes:

- `human_ai_scale`: RAID-reference scale on a `0 -> human` to `100 -> ai` axis, anchored so `50` is the score boundary

To preserve machine-readable behavior for ad hoc text scoring:

```bash
fast-ai-detector --text "This is a test." --json
```

Optional vector-space SAE analysis for `unsupervised` mode:

```bash
fast-ai-detector --text "This is a test." --explain-sae --sae-top-k 5
```

This prints the score table, then a blank line, then a TSV feature table with:

- `state_vs_midpoint`: signed loading for how this example differs from the RAID midpoint on that SAE feature
- `usual_assoc`: whether that feature is usually AI-associated or human-associated under the detector direction
- `ai_net_push`: signed push on the detector in AI space for this example on that feature

The default CLI output shows explained Neuronpedia features only. Unexplained features are suppressed from the default lists.

Use `--json` with `--text` if you want the SAE explanation as structured JSON instead of tables.

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

Optional vector-space SAE explanations for file mode go to a JSONL sidecar:

```bash
fast-ai-detector \
  --input rows.csv \
  --output scored.csv \
  --explain-sae \
  --sae-top-k 5 \
  --sae-output-jsonl scored.sae.jsonl
```

This keeps the CSV flat while writing one structured explanation object per row into the JSONL file.

## Notes

- `--device auto` uses CUDA if available, otherwise CPU.
- Positive scores indicate stronger evidence for AI in both modes.
- `human_ai_scale` is not a probability. It is a signed reference scale on the detector score axis:
  `0 = strongly human side`, `50 = near boundary`, `100 = strongly AI side`.
- `unsupervised` downloads `unignorant/fast-ai-detector`.
- `raid-finetune` downloads `unignorant/fast-ai-detector-raid-finetune`.
- `--explain-sae` is only supported for `unsupervised`.
- The SAE analysis is a decoder-side vector-space interpretation of the example's position relative to the RAID midpoint in SAE space, annotated by detector association. It is not a probability or exact causal attribution.
