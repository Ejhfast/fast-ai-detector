# fast-ai-detector

`fast-ai-detector` is a fast local CLI for scoring text as likely human- or AI-written with compact distilled models that run on CPU or GPU. The package has two modes:

- `unsupervised` (default): a contrast detector with optional SAE-based document feature inspection (contrast vectors computed on RAID dataset)
- `raid-finetune`: a stronger fully supervised classifier head on top of the same distilled backbone (trained on RAID dataset)

What makes it unusual is the combination of small size and interpretability. It uses a small (40M param) distilled student model that approximates mean-pooled residual representations from a larger (4B param) Gemma model. In `unsupervised` mode, those residual-style outputs can also be annotated with SAE features from the teacher model's interpretability stack.

Current reference numbers:

- RAID held-out validation (held out 20% from train), `raid-finetune`: ROC-AUC `0.9958`, AP `0.99987`, TPR at `1%` FPR `0.9188`, TPR at `5%` FPR `0.9801`
- Pangram benchmark, default `unsupervised`: accuracy `0.8841`, ROC-AUC `0.9423`, AP `0.9469`

The bundled benchmarks here are useful sanity checks, but they are not rich in text from the newest model families. In particular, they contain little or no output from GPT-5-era systems and later, so you should not expect these scores to transfer unchanged to the latest model outputs.

## Installation

```bash
pip install -e .
```

On first use, the selected model bundle is downloaded from Hugging Face and cached locally.

## Scoring

### Direct text

Default output is a compact human-readable table. These are real examples from `examples/pangram_benchmark.csv`.

Human-like Pangram review:

```bash
fast-ai-detector --text "Went there 3 weeks ago, the place was jammed. Service was great food (breakfast) was excellent. We will be going back."
```

```text
mode          label  score       human_ai_scale
unsupervised  human  -75.602432  38.080172
```

AI-like Pangram review:

```bash
fast-ai-detector --text "I love these stories. The characters are complex and relatable, and the plot twists keep me on the edge of my seat. The writing style is engaging and descriptive, making it easy to immerse myself in the world of the story. Each tale is unique and captivating, and I find myself thinking about them long after I've finished reading. I highly recommend these stories to anyone looking for thought-provoking and entertaining reads."
```

```text
mode          label  score       human_ai_scale
unsupervised  ai     423.319458  94.671712
```

`human_ai_scale` is a RAID-reference scale, not a probability:

- `0` = strongly human side
- `50` = near the decision boundary
- `100` = strongly AI side

If you want the RAID-specific finetune model instead of the default contrast model:

```bash
fast-ai-detector --mode raid-finetune --device cuda --text "Your text here"
```

For machine-readable direct output:

```bash
fast-ai-detector --text "Your text here" --json
```

### CSV / TSV

The repo includes `examples/pangram_benchmark.csv` as a ready-to-run example dataset.

```bash
fast-ai-detector \
  --input examples/pangram_benchmark.csv \
  --text-column text \
  --output examples/pangram_benchmark_scored.csv
```

The tool preserves the original columns and appends:

- `fast_ai_detector_label`
- `fast_ai_detector_score`
- `fast_ai_detector_human_ai_scale`

Example output rows:

```csv
text,label,tags,fast_ai_detector_label,fast_ai_detector_score,fast_ai_detector_human_ai_scale
"Went there 3 weeks ago, the place was jammed. Service was great food (breakfast) was excellent. We will be going back.",0,"['reviews', None]",human,-75.602432,38.080172
"I love these stories. The characters are complex and relatable, and the plot twists keep me on the edge of my seat. The writing style is engaging and descriptive, making it easy to immerse myself in the world of the story. Each tale is unique and captivating, and I find myself thinking about them long after I've finished reading. I highly recommend these stories to anyone looking for thought-provoking and entertaining reads.",1,"['reviews', 'gpt-3.5-turbo-1106']",ai,423.319458,94.671712
```

## SAE

`unsupervised` mode can also expose document-level SAE annotations derived from the Gemma interpretability stack the student was distilled from.

```bash
fast-ai-detector \
  --text "Subject: Exciting New Classes Announcement Dear Valued Students, We are thrilled to announce the launch of our new class offerings! From advanced coding courses to creative writing workshops, there's something for everyone. Register now to secure your spot and embark on a new learning journey. Join us in exploring your passions and expanding your horizons. Let's grow together! Best, Your Team of Dedicated Educators." \
  --explain-sae \
  --sae-top-k 5
```

```text
mode          label  score       human_ai_scale
unsupervised  ai     328.589661  84.249271

feature_index  title                              state_vs_midpoint  usual_assoc  ai_net_push
942            categories and definitions         19.829247          ai           170.849304
1310           improvements and explanations      17.385456          ai           115.146446
7748           code snippets                      17.301167          ai           120.488045
3341           struggling with                    16.845036          ai           54.685085
7938           Corporate, data, September, items  12.203426          ai           54.555244
```

These annotations are intended as document fingerprints in SAE space, not as calibrated probabilities or exact causal attributions.

## Approach

This project grew out of experiments on whether much smaller models could approximate mean-pooled residual representations from larger LLMs closely enough to preserve useful downstream geometry. The resulting detector uses a compact 4-layer student trained to predict a Gemma layer-17 mean-pooled residual representation.

From there, two detector variants were built on top of the student:

- `unsupervised`: a contrast direction learned in the student's residual space
- `raid-finetune`: a supervised classifier head trained for the RAID benchmark

The main motivation for doing this at all was not just speed. If a small model can stay close enough to the teacher representation, then some of the interpretability infrastructure built around the teacher model can still be reused. That is what powers the optional SAE annotations in `unsupervised` mode: the small model is fast enough for local use, but the outputs can still be inspected with the teacher's SAE dictionary.
