# wildfire-spread-prediction
Mini Project
# Efficient Deep Learning for Next-Day Wildfire Spread Prediction

CSS7102 – Mini Project | Presidency School of Artificial Intelligence & Advanced Computing, Presidency University

**Problem Statement:** PSAIAC_364 — *ML-Based Wildfire Spread Prediction* (Machine Learning specialization)

**SDG Mapping:** SDG 13 (Climate Action) · SDG 15 (Life on Land) · SDG 11 (Sustainable Cities & Communities)

Overview

Wildfires are growing more frequent and severe due to climate change, land-use pressure, and prolonged drought. This project builds a deep learning model that predicts **next-day wildfire spread** from satellite-derived weather, vegetation, topography, and prior fire-extent data, and pairs that prediction with a lightweight, uncertainty-aware decision-support layer — a four-tier risk classification (Low / Moderate / High / Extreme) intended to support faster evacuation and containment decisions.

The core contribution is an **efficiency-focused re-architecture** of a recently published attention-based baseline: we reproduce a 2025/2026 IEEE Attention U-Net as the benchmark, then design a lighter model (depthwise-separable convolutions + linear attention) that targets comparable or better accuracy at a fraction of the parameters, FLOPs, and inference latency — a gap we identified as unresolved in the current literature (see [Research Gaps](#research-gaps) below).

## Dataset

**Primary — [Next Day Wildfire Spread (NDWS)](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)** (Huot et al., 2022)
~18,500 samples, 64×64 px, 12 channels (weather, vegetation, topography, drought index, population, prior fire mask). Available on Kaggle and TensorFlow Datasets.

**Secondary (stretch goal, generalization test) — [WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS)** (Gerard et al., 2023)
13,607 images across 607 fire events (2018–2021), 23 channels, full daily time series. Used only to evaluate cross-dataset generalization if time permits.

## Base Paper

> *"Predicting Wildfire Spread with Attention U-Net: A Multi-Feature Geospatial Deep Learning Approach,"* IEEE Xplore Document 11471469, 2025/2026.

An Attention U-Net that segments next-day fire spread from 12 geospatial input channels using attention gates. We reproduce this as our benchmark and compare our lightweight variant against it directly on accuracy and efficiency metrics.

## Research Gaps

A review of 17 papers (2022–2026) on wildfire spread prediction, cross-referenced against two systematic/bibliometric reviews of the field, surfaced five open gaps:

1. **Unresolved efficiency–accuracy trade-off** — lightweight models sacrifice accuracy vs. full-size SOTA; no paper reports params/FLOPs/latency together with accuracy against a common baseline.
2. **Reproducibility gap** — only 7.7% of wildfire ML/DL studies release public code.
3. **No efficient model is paired with uncertainty quantification**, even though confidence matters most for downstream decisions like evacuation.
4. **Single-dataset, single-region evaluation** — cross-dataset/cross-event generalization is rarely tested.
5. **Limited translation from raw prediction to decision support** — most papers stop at a pixel-wise probability map.

This project targets gaps 1, 2, 3, and 5 directly, with gap 4 as a stretch goal (see the full literature review document for details).

## Proposed Approach

```
Input Data          Preprocessing        Lightweight Model         Spread Prediction        Uncertainty + Risk Tier
(NDWS: weather,  →   (Normalization,  →   (Depthwise-separable  →  (Next-day fire spread →  (MC-Dropout confidence →
vegetation,          train/val/test       U-Net + linear           probability map)          Low/Med/High/Extreme)
topography,          split)               attention)
prior fire mask)
```

1. **Core prediction model** — lightweight U-Net encoder with depthwise-separable convolutions and a linear-attention block, benchmarked against the base paper's Attention U-Net on AUC-PR, F1, IoU, parameter count, FLOPs, and CPU/GPU inference latency.
2. **Decision-support AI layer** — a Monte Carlo Dropout confidence estimator converts the pixel-wise spread probability into a rule-based risk tier with a recommended-response note (the project's explicit "intelligent decision-making" component).
3. **Stretch goal** — zero-shot evaluation on WildfireSpreadTS or a 2023 Maui fires case study to report a generalization gap.

## Tech Stack

- Python 3.10+
- PyTorch — model development
- NumPy, Pandas — data handling and preprocessing
- TensorFlow Datasets / Kaggle API — NDWS dataset access
- scikit-learn — AUC-PR, F1, IoU evaluation metrics
- Matplotlib / Seaborn — visualization of results and spread maps
- Git & GitHub — version control
- Google Colab / Jupyter Notebook — GPU-backed development environment

## Repository Structure

```
.
├── data/                 # dataset download/preprocessing scripts (not committed — see .gitignore)
├── notebooks/            # exploratory notebooks
├── src/
│   ├── models/           # base paper reproduction + lightweight model architectures
│   ├── training/         # training loops, config
│   ├── evaluation/       # metrics, efficiency benchmarking, uncertainty module
│   └── utils/            # data loaders, preprocessing
├── reports/              # literature review, proposal, review slide decks
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/wildfire-spread-prediction.git
cd wildfire-spread-prediction
pip install -r requirements.txt
```

Dataset setup (NDWS via Kaggle):

```bash
kaggle datasets download -d fantineh/next-day-wildfire-spread -p data/
unzip data/next-day-wildfire-spread.zip -d data/ndws
```

## Project Timeline

| Phase | Weeks |
|---|---|
| Literature review & topic finalization | 1 |
| Dataset acquisition & preprocessing | 2–3 |
| Reproduce base paper (Attention U-Net) | 3–4 |
| Design lightweight attention architecture | 4–5 |
| Training & hyperparameter tuning | 6–7 |
| Efficiency benchmarking | 7 |
| MC-Dropout uncertainty + risk-tier module | 8–9 |
| Report writing & Review-2 prep | 9–10 |
| Final testing & paper drafting | 10–11 |
| Review-3 prep & submission | 11 |

## References

1. F. Huot, R. L. Hu, N. Goyal, T. Sankar, M. Ihme, and Y.-F. Chen, "Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, 2022.
2. S. Gerard, Y. Zhao, and J. Sullivan, "WildfireSpreadTS: A Dataset of Multi-Modal Time Series for Wildfire Spread Prediction," *NeurIPS Datasets and Benchmarks Track*, 2023.
3. "Predicting Wildfire Spread with Attention U-Net: A Multi-Feature Geospatial Deep Learning Approach," IEEE Xplore Document 11471469, 2025/2026. *(base paper)*
4. S. Lahrichi, J. Bova, J. Johnson, and J. Malof, "Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark," *IEEE/CVF WACV*, 2026.
5. "LinU-Mamba: Visual Mamba U-Net with Linear Attention to Predict Wildfire Spread," *Remote Sensing (MDPI)*, vol. 17, no. 15, art. 2715, 2025.
6. E. Meco, Y. Luo, E. Hamdan, A. Watts, and A. E. Cetin, "ShearFuse-UNet: Hadamard, DCT, and Shearlet Transform Fusion for Next-Day Wildfire Spread Prediction," arXiv:2606.14071, 2026.
7. H. S. Andrianarivony and M. A. Akhloufi, "Machine Learning and Deep Learning for Wildfire Spread Prediction: A Review," *Fire (MDPI)*, vol. 7, no. 12, art. 482, 2024.

*Full literature review with all 17 papers and the detailed gap analysis is in `reports/`.*

## Team

| Roll Number | Name |
|---|---|
| 20231CCS0001 | Abhinav Harisree |
| 20231CCS0013 | Hashir Ashraf |
| 20231CCS0044 | Gautham Raj P |
| 20231CCS0062 | Kritik V |

**Guide:** Mr. Harikrishnan N
Professor
School of Computer Science and Engineering
Presidency University

**Academic mini-project — Presidency University, CSS7102.**
