# fast-ai-detector

`fast-ai-detector` is a fast local CLI for scoring text as likely AI-written with models that run on CPU or GPU. 

The core model is a small distilled transformer (~40M param) that approximates mean-pooled residual representations from a larger (4B param) Gemma model. In `contrast` mode, those residual-style outputs can also be annotated with SAE features from the teacher model's interpretability stack. Contrast vectors were computed from the [RAID](https://raid-bench.xyz/) training dataset.

Modes:
- `contrast` (default): contrast-based prediction and SAE-based document feature inspection
- `raid-finetune`: the core transformer model finetuned with a classifier head (trained on the RAID dataset)

Current reference numbers:

| Dataset | Mode | Balanced Accuracy | ROC-AUC | TPR @ 5% FPR |
| --- | --- | ---: | ---: | ---: |
| RAID held-out validation | `contrast` | `0.8078` | `0.9343` | `0.7637` |
| RAID held-out validation | `raid-finetune` | `0.9642` | `0.9958` | `0.9801` |
| Pangram benchmark | `contrast` | `0.8827` | `0.9425` | `0.7856` |
| Pangram benchmark | `raid-finetune` | `0.6731` | `0.8993` | `0.6466` |

The contrast model (default) is recommended for normal use as the raid-finetuned model seems poorly calibrated around the human/ai threshold for text that falls outside the distribution of the RAID training data. 

The pangram benchmark is small and included in this repo for reproducability. 

Note: the bundled benchmarks contain little or no output from GPT-5-era systems and later, so you should not expect these scores to transfer unchanged to the latest model outputs.

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
mode      label  score       human_ai_scale
contrast  human  -75.602432  38.080172
```

AI-like Pangram review:

```bash
fast-ai-detector --text "I love these stories. The characters are complex and relatable, and the plot twists keep me on the edge of my seat. The writing style is engaging and descriptive, making it easy to immerse myself in the world of the story. Each tale is unique and captivating, and I find myself thinking about them long after I've finished reading. I highly recommend these stories to anyone looking for thought-provoking and entertaining reads."
```

```text
mode      label  score       human_ai_scale
contrast  ai     423.319458  94.671712
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

`contrast` mode can also expose document-level SAE annotations derived from the Gemma interpretability stack the student was distilled from.

```bash
fast-ai-detector \
  --text "Subject: Exciting New Classes Announcement Dear Valued Students, We are thrilled to announce the launch of our new class offerings! From advanced coding courses to creative writing workshops, there's something for everyone. Register now to secure your spot and embark on a new learning journey. Join us in exploring your passions and expanding your horizons. Let's grow together! Best, Your Team of Dedicated Educators." \
  --explain-sae \
  --sae-top-k 5
```

```text
mode      label  score       human_ai_scale
contrast  ai     328.589661  84.249271

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

- `contrast`: a contrast direction learned in the student's residual space
- `raid-finetune`: a supervised classifier head trained for the RAID benchmark

The main motivation for doing this at all was not just speed. If a small model can stay close enough to the teacher representation, then some of the interpretability infrastructure built around the teacher model can still be reused. That is what powers the optional SAE annotations in `contrast` mode: the small model is fast enough for local use, but the outputs can still be inspected with the teacher's SAE dictionary.
