# =============================================================================
# Colab Quickstart -- paste each `# %%` cell into a separate Colab cell
# (or File > Upload notebook after renaming this to .ipynb; Colab also opens
# plain .py files with # %% markers as notebooks via "Open with Colaboratory").
#
# Runtime > Change runtime type > T4 GPU, before running.
# =============================================================================

# %% [1] Clone your repo and install dependencies
# !git clone https://github.com/<your-username>/wildfire-spread-prediction.git
# %cd wildfire-spread-prediction
# !pip install -q -r requirements.txt

# %% [2] Get your Kaggle API token onto this machine
# 1. kaggle.com -> your profile picture -> Settings -> API -> "Create New Token"
#    (downloads kaggle.json to your computer)
# 2. Run this cell, click "Choose Files", upload that kaggle.json
from google.colab import files
uploaded = files.upload()  # select kaggle.json

# %% [3] Put the token where the kaggle CLI expects it, then download NDWS
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d fantineh/next-day-wildfire-spread -p data/raw
# !unzip -q data/raw/next-day-wildfire-spread.zip -d data/raw

# %% [4] Sanity-check the data pipeline
# !python -m src.data --data_dir data/raw

# %% [5] Train the baseline (reproduces the base paper's Attention U-Net)
# !python -m src.training.train --model attention --data_dir data/raw --epochs 30

# %% [6] Train your lightweight model
# !python -m src.training.train --model lightweight --data_dir data/raw --epochs 30

# %% [7] Evaluate both and get the feasibility-slide numbers
# !python -m src.evaluation.evaluate --model attention --ckpt checkpoints/attention_best.pt --data_dir data/raw
# !python -m src.evaluation.evaluate --model lightweight --ckpt checkpoints/lightweight_best.pt --data_dir data/raw

# %% [8] Try the uncertainty + risk-tier module on a real trained model
import torch
from src.models.lightweight_unet import LightweightUNet
from src.uncertainty import mc_dropout_predict, risk_tier_report
from src.data import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightUNet(in_channels=12, base_ch=24, dropout_p=0.2).to(device)
state = torch.load("checkpoints/lightweight_best.pt", map_location=device)
model.load_state_dict(state["model_state"])

_, _, test_loader = get_dataloaders("data/raw", batch_size=1)
x, y_true = next(iter(test_loader))
x = x.to(device)

mean, std = mc_dropout_predict(model, x, n_samples=20)
report = risk_tier_report(mean.cpu().numpy(), std.cpu().numpy())
print(report)
