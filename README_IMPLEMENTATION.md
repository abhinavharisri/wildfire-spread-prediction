# Implementation Guide

Everything below is written and ready. This file is the exact sequence to
go from zero to trained models + real feasibility-slide numbers.

## Why this needs to run on Colab, not "in chat"

Training needs a GPU and a real ~2-3GB dataset download authenticated with
your own Kaggle account — neither is something that can happen inside this
conversation. Google Colab's free T4 GPU is exactly what's listed on your
Review-1 hardware requirements slide, so that's the intended runtime.

## What's already done (verified without needing to run anything)

The parameter-efficiency claim — the core of your Gap-1 contribution — is
real and already checked, computed directly from the layer definitions in
the code (see the math in the file, not an estimate):

| Model | Parameters |
|---|---|
| AttentionUNet (base paper reproduction) | 7,853,881 |
| LightweightUNet (yours) | 1,447,957 |
| **Reduction** | **81.6% fewer (5.42x smaller)** |

What's *not* yet done, because it requires a GPU and the real dataset:
accuracy (AUC-PR/F1/IoU), FLOPs, latency, and the trained uncertainty/risk-
tier outputs. Steps 1-7 below get you those.

## Step-by-step

**1. Push this folder to your GitHub repo** (the one from your Review-1
   deck). `git add . && git commit -m "Initial implementation" && git push`

**2. Open Google Colab** ([colab.research.google.com](https://colab.research.google.com)) → File → Open notebook → GitHub tab → paste your repo URL → open `notebooks/colab_quickstart.py`. Colab will offer to convert it to a notebook — accept.

**3. Runtime → Change runtime type → T4 GPU → Save.**

**4. Get a Kaggle API token** (one-time, ~1 minute):
   - Go to kaggle.com → your profile picture → *Settings* → *API* → *Create New Token*. This downloads `kaggle.json`.
   - **Do not paste your Kaggle credentials into this chat or any chat** — upload the file directly in the Colab cell provided (cell [2] in the quickstart), which is the one place it should go.

**5. Run the cells in order** (uncomment the `!` shell lines as you go — they're commented out so the file can be read as reference first):
   - Cell 1: clone + install deps (~2 min)
   - Cell 2-3: upload `kaggle.json`, download + unzip NDWS (~5-10 min, ~2GB)
   - Cell 4: sanity-check the data loader — confirms shapes are `[B, 12, 64, 64]` / `[B, 1, 64, 64]`
   - Cell 5: train the baseline Attention U-Net (~30-45 min on a T4 for 30 epochs)
   - Cell 6: train your lightweight model (should be noticeably faster per epoch — that's your speed number)
   - Cell 7: evaluate both — prints AUC-PR, F1, IoU, parameters, FLOPs, latency for each, formatted to paste straight into the feasibility slide
   - Cell 8: run MC-Dropout + risk-tier on a real prediction

**6. Save the printed metrics** from cell 7 somewhere (a plain text file in the repo is fine) — those are your real "Already demonstrated / Performance achieved / Speed" numbers for the report and the Review-2 deck.

## Repo layout reference

```
src/data.py                        # Module 1: Data Preprocessing
src/models/attention_unet.py       # Module 2: Baseline Reproduction
src/models/lightweight_unet.py     # Module 3: Lightweight Model
src/training/train.py              # Module 4: Training & Tuning
src/evaluation/evaluate.py         # Module 5 + 8: Efficiency Benchmarking + Evaluation
src/uncertainty.py                 # Module 6 + 7: Uncertainty Estimation + Risk Triage
notebooks/colab_quickstart.py      # run this in Colab
```

## If something breaks

- **`FileNotFoundError: No TFRecord shards found`** — the Kaggle download/unzip in cell 3 didn't complete; check `data/raw/` actually has files matching `*train*.tfrecord*`.
- **CUDA out of memory during training** — lower `--batch_size` (try 16 or 8) in cells 5/6.
- **`kaggle: command not found`** — cell 1's `pip install -r requirements.txt` includes the `kaggle` package; re-run it.
- **Training loss is `nan`** — lower `--lr` (try `1e-4`) in cells 5/6.
