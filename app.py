"""
Wildfire Spread Prediction - live demo
======================================
    pip install -r requirements-demo.txt
    python -m streamlit run app.py

Needs the `demo/` folder produced by `python -m src.export_demo` (metrics.json,
demo_bundle.npz and, for live inference, the two .pt checkpoints). The 2 GB NDWS
dataset and a GPU are NOT needed: the lightweight model runs on CPU in milliseconds.
"""
import io
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.uncertainty import RISK_TIERS, RiskThresholds, assign_risk_tier, risk_tier_report

DEMO_DIR = os.environ.get("WILDFIRE_DEMO_DIR", "demo")

# ------------------------------------------------------------------ palette
PAPER, PAPER2, INK, INK2, RULE = "#f6f2ea", "#ece6d9", "#1d1b18", "#55504a", "#d8d0bf"
EMBER, SLATE = "#c2410c", "#3b3d40"
TIER_COLORS = ["#7d9368", "#d6a23a", "#c2602a", "#7f1d1d"]
EMBER_CMAP = LinearSegmentedColormap.from_list("ember", ["#f0eade", "#f6cd96", "#ea803c", "#ac3012", "#3a0a08"])
SLATE_CMAP = LinearSegmentedColormap.from_list("slate", ["#f0eade", "#aababa", "#567882", "#1e3846"])
DIFF_CMAP = LinearSegmentedColormap.from_list("diff", ["#3b6a8a", "#f0eade", "#b4310d"])
MASK_CMAP = ListedColormap([PAPER2, "#b4310d"])
TIER_CMAP = ListedColormap([PAPER2] + TIER_COLORS[1:])
ERR_COLORS = [PAPER2, "#2f6f4f", "#d6a23a", "#7f1d1d"]  # true-negative, TP, FP, FN
ERR_CMAP = ListedColormap(ERR_COLORS)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": RULE, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

st.set_page_config(page_title="Wildfire Spread Prediction", layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{ background:{PAPER}; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  #MainMenu, footer, [data-testid="stDecoration"] {{ visibility:hidden; height:0; }}
  [data-testid="stExpandSidebarButton"] {{ visibility:visible !important; }}
  .block-container {{ max-width:1180px; padding-top:1.4rem; padding-bottom:4rem; }}
  [data-testid="stSidebar"] {{ background:{PAPER2}; border-right:1px solid {RULE}; }}
  [data-testid="stSidebarCollapseButton"] {{ display:none !important; }}
  [data-testid="stSidebarHeader"] {{ height:.6rem; min-height:0; padding:0; }}
  [data-testid="stSidebarUserContent"] {{ padding-top:.4rem; }}
  [data-testid="stSidebar"] .stButton button {{ width:100%; font-family:ui-monospace,Menlo,monospace; font-size:.74rem;
      letter-spacing:.06em; text-transform:uppercase; border:1px solid {INK}; background:{PAPER}; color:{INK}; border-radius:2px; padding:.35rem .4rem; }}
  [data-testid="stSidebar"] .stButton button:hover {{ background:{INK}; color:{PAPER}; border-color:{INK}; }}
  [data-testid="stSidebar"] .stButton button:disabled {{ opacity:.45; }}
  [data-testid="stSidebar"] label p {{ font-size:.92rem; }}
  [data-testid="stSidebar"] .brand {{ font-size:1.15rem; line-height:1.2; padding:.2rem 0 .7rem; border-bottom:2px solid {INK}; margin-bottom:.4rem; }}
  [data-testid="stSidebar"] .brand small {{ display:block; font-family:ui-monospace,Menlo,monospace; font-size:.66rem;
      letter-spacing:.14em; text-transform:uppercase; color:{INK2}; margin-bottom:.25rem; }}
  [data-testid="stSidebar"] [data-testid="stExpander"] {{ border:1px solid {RULE}; background:{PAPER}; border-radius:2px; }}
  [data-testid="stSidebar"] [data-testid="stExpander"] summary {{ font-family:ui-monospace,Menlo,monospace; font-size:.74rem;
      letter-spacing:.1em; text-transform:uppercase; }}
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ font-family:ui-monospace,Menlo,monospace; font-size:.78rem;
      letter-spacing:.14em; text-transform:uppercase; font-weight:500; border-bottom:1px solid {INK}; padding-bottom:.4rem; }}
  html, body, p, li, label, div {{ font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; }}
  .eyebrow {{ font-family:ui-monospace,Menlo,monospace; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:{INK2}; }}
  .mast {{ border-bottom:2px solid {INK}; padding:1.2rem 0 1.3rem; margin-bottom:.8rem; }}
  .mast h1 {{ font-weight:500; font-size:2.9rem; line-height:1.05; letter-spacing:-.015em; margin:.7rem 0 .6rem; padding:0; }}
  .mast h1 em {{ color:{EMBER}; }}
  .mast p {{ color:{INK2}; font-size:1.1rem; max-width:46rem; margin:0; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:1.6rem; border-bottom:1px solid {RULE}; flex-wrap:wrap; }}
  .stTabs [data-baseweb="tab"] {{ font-family:ui-monospace,Menlo,monospace; font-size:.78rem; letter-spacing:.12em;
      text-transform:uppercase; padding-left:0; padding-right:0; }}
  .stTabs [aria-selected="true"] {{ color:{EMBER}; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background:{EMBER}; }}
  h2, h3 {{ font-weight:500; letter-spacing:-.01em; }}
  .glance {{ display:grid; grid-template-columns:repeat(4,1fr); border-top:1px solid {INK}; border-bottom:1px solid {RULE}; margin:.4rem 0 1.2rem; }}
  .glance > div {{ padding:.9rem 1rem 1rem 0; border-right:1px solid {RULE}; margin-right:1rem; }}
  .glance > div:last-child {{ border-right:none; margin-right:0; }}
  .glance .v {{ font-size:2.8rem; line-height:1; letter-spacing:-.02em; }}
  .glance .v small {{ font-size:.5em; color:{EMBER}; }}
  .glance .l {{ font-family:ui-monospace,Menlo,monospace; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; margin-top:.5rem; }}
  .glance .d {{ font-size:.88rem; color:{INK2}; margin-top:.15rem; }}
  .tierrow {{ display:grid; grid-template-columns:5.5rem 1fr 3.4rem; gap:.6rem; align-items:center;
      font-family:ui-monospace,Menlo,monospace; font-size:.78rem; margin:.35rem 0; }}
  .tierrow .t {{ height:9px; }}
  .quote {{ margin:1rem 0; padding:.1rem 0 .1rem .9rem; border-left:3px solid {EMBER}; font-size:1.2rem; font-style:italic; line-height:1.35; }}
  .bigtier {{ font-size:2.4rem; line-height:1; letter-spacing:-.01em; }}
  .chip {{ display:inline-block; width:.95rem; height:.95rem; margin-right:.5rem; vertical-align:-.1rem; }}
  .cap {{ font-size:.86rem; color:{INK2}; border-top:1px solid {RULE}; padding-top:.4rem; margin-top:.2rem; }}
  .src {{ font-family:ui-monospace,Menlo,monospace; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:{EMBER}; }}
  [data-testid="stDataFrame"] {{ border:none; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# If the browser remembers the sidebar as collapsed, reopen it once per page load.
import streamlit.components.v1 as components
components.html("""<script>
try {
  const p = window.parent;
  Object.keys(p.localStorage).filter(k => k.startsWith('stSidebarCollapsed')).forEach(k => p.localStorage.removeItem(k));
  const open = () => { const b = p.document.querySelector('[data-testid="stExpandSidebarButton"]'); if (b) b.click(); };
  open(); setTimeout(open, 600); setTimeout(open, 1500);
} catch (e) {}
</script>""", height=0)

CHANNELS = ["Elevation", "Wind direction", "Wind speed", "Min temperature", "Max temperature", "Humidity",
            "Precipitation", "Drought index", "Vegetation (NDVI)", "Population", "Energy release", "Fire today"]
STAGE_NAMES = ["enc1", "enc2", "enc3", "enc4", "bottleneck", "bottleneck_attn",
               "dec4", "dec3", "dec2", "dec1", "out_conv"]
STAGE_ROLE = {"enc1": "encoder: compress", "enc2": "encoder: compress", "enc3": "encoder: compress",
              "enc4": "encoder: compress", "bottleneck": "bottleneck conv",
              "bottleneck_attn": "linear attention (global context)", "dec4": "decoder: expand + skip",
              "dec3": "decoder: expand + skip", "dec2": "decoder: expand + skip",
              "dec1": "decoder: expand + skip", "out_conv": "1x1 conv to fire probability"}
MODES = ["Live (run models now)", "Precomputed"]
# Offsets are in normalised (z-score) units; PDSI is negative in drought, so drier = lower.
PRESETS = {
    "Hotter, drier, windier": {"Max temperature": 1.5, "Min temperature": 1.0, "Humidity": -1.5,
                               "Drought index": -1.5, "Wind speed": 1.5},
    "Cooler and wetter": {"Max temperature": -1.5, "Min temperature": -1.0, "Humidity": 1.5,
                          "Drought index": 1.5, "Precipitation": 1.5, "Wind speed": -1.0},
}


# ----------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def load_bundle():
    path = os.path.join(DEMO_DIR, "demo_bundle.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


@st.cache_data(show_spinner=False)
def load_metrics():
    path = os.path.join(DEMO_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models():
    """Returns (attention, lightweight, error_message). Models are None if torch/checkpoints are missing."""
    try:
        import torch
        from src.models.attention_unet import AttentionUNet
        from src.models.lightweight_unet import LightweightUNet
        a_path = os.path.join(DEMO_DIR, "attention_best.pt")
        l_path = os.path.join(DEMO_DIR, "lightweight_best.pt")
        if not (os.path.exists(a_path) and os.path.exists(l_path)):
            return None, None, "checkpoints not found in demo/"
        attn = AttentionUNet(in_channels=12, base_ch=32)
        attn.load_state_dict(torch.load(a_path, map_location="cpu")["model_state"])
        attn.eval()
        light = LightweightUNet(in_channels=12, base_ch=24, dropout_p=0.2)
        light.load_state_dict(torch.load(l_path, map_location="cpu")["model_state"])
        light.eval()
        return attn, light, ""
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"


# --------------------------------------------------- cached model computations
@st.cache_data(show_spinner=False, max_entries=64)
def det_predict(_model, x, demo_dir, tag):
    """Deterministic (dropout off) probability map, plus wall time in ms."""
    import torch
    t0 = time.perf_counter()
    _model.eval()
    with torch.no_grad():
        out = _model(torch.from_numpy(x[None].astype(np.float32))).numpy()[0, 0]
    return out, 1000 * (time.perf_counter() - t0)


@st.cache_data(show_spinner=False, max_entries=64)
def mc_predict(_light, x, n, seed, demo_dir):
    """n stochastic passes with dropout on. Same maths as src.uncertainty.mc_dropout_predict,
    but keeps the first few passes so the 'Under the hood' tab can show them."""
    import torch
    torch.manual_seed(int(seed))
    xt = torch.from_numpy(x[None].astype(np.float32))
    t0 = time.perf_counter()
    _light.enable_mc_dropout()
    with torch.no_grad():
        stack = np.stack([_light(xt).numpy()[0, 0] for _ in range(int(n))])
    _light.eval()
    return {"mean": stack.mean(0), "std": stack.std(0, ddof=1), "samples": stack[:5].copy(),
            "ms": 1000 * (time.perf_counter() - t0)}


@st.cache_data(show_spinner=False, max_entries=16)
def stage_shapes(_light, x, demo_dir):
    import torch
    rows, handles = [], []
    t0 = [0.0]

    def make_hook(name):
        def hook(_m, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            rows.append((name, tuple(o.shape), 1000 * (time.perf_counter() - t0[0])))
        return hook

    for nm in STAGE_NAMES:
        if hasattr(_light, nm):
            handles.append(getattr(_light, nm).register_forward_hook(make_hook(nm)))
    _light.eval()
    try:
        with torch.no_grad():
            t0[0] = time.perf_counter()
            _light(torch.from_numpy(x[None].astype(np.float32)))
    finally:
        for h in handles:
            h.remove()
    return rows


# ------------------------------------------------------------------ helpers
def fig_png(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def style_image_axes(axes):
    for ax in np.ravel(axes):
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(INK); sp.set_linewidth(0.8)


def patch_scores(pred, true, thr=0.5):
    b = pred >= thr
    t = true >= 0.5
    tp = np.logical_and(b, t).sum()
    fp = np.logical_and(b, ~t).sum()
    fn = np.logical_and(~b, t).sum()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return f1, iou


def error_map(pred, true, thr):
    b, t = pred >= thr, true >= 0.5
    m = np.zeros(pred.shape, dtype=np.int64)
    m[b & t] = 1   # true positive
    m[b & ~t] = 2  # false alarm
    m[~b & t] = 3  # missed fire
    return m


def glance(cards):
    html = "".join(f'<div><div class="v">{v}<small>{u}</small></div><div class="l">{l}</div><div class="d">{d}</div></div>'
                   for v, u, l, d in cards)
    st.markdown(f'<div class="glance">{html}</div>', unsafe_allow_html=True)


def fnum(v, d=3):
    return "n/a" if v is None or v != v else f"{v:.{d}f}"


def legend(items):
    return " &nbsp; ".join(f'<span class="chip" style="background:{c}"></span>{n}' for n, c in items)


@st.cache_data(show_spinner=False)
def chart_png(kind, demo_dir):
    m = load_metrics()
    fig, ax = plt.subplots(figsize=(5.4, 4))
    if kind == "pr":
        for name, key, color in (("Attention U-Net", "attention", SLATE), ("Lightweight U-Net", "lightweight", EMBER)):
            c = m["pr_curves"][key]
            ax.plot(c["recall"], c["precision"], label=name, color=color, lw=2)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-recall, all test pixels", loc="left", fontsize=10)
    else:
        s = m["f1_sweep"]
        ax.plot(s["thresholds"], s["attention"], label="Attention U-Net", color=SLATE, lw=2)
        ax.plot(s["thresholds"], s["lightweight"], label="Lightweight U-Net", color=EMBER, lw=2)
        ax.set_xlabel("Decision threshold"); ax.set_ylabel("F1")
        ax.set_title("F1 against threshold", loc="left", fontsize=10)
    ax.legend(frameon=False)
    return fig_png(fig, 110)


@st.cache_data(show_spinner=False, max_entries=32)
def inputs_png(x):
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.3))
    for k, ax in enumerate(axes.ravel()):
        if k == 11:
            ax.imshow(x[k] > 0, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(x[k], cmap=EMBER_CMAP, interpolation="nearest")
        ax.set_title(CHANNELS[k], fontsize=8.5, loc="left", color=INK2, family="monospace")
    style_image_axes(axes)
    fig.tight_layout()
    return fig_png(fig, 110)


@st.cache_data(show_spinner=False)
def params_png(_attn, _light, demo_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (title, mdl, color) in zip(axes, (("Attention U-Net", _attn, SLATE), ("Lightweight U-Net", _light, EMBER))):
        parts = [(n, sum(p.numel() for p in m.parameters())) for n, m in mdl.named_children()]
        parts = [(n, c) for n, c in parts if c > 0]
        ax.barh([n for n, _ in parts][::-1], [c / 1e6 for _, c in parts][::-1], color=color)
        ax.set_xlabel("Parameters (millions)")
        ax.set_title(f"{title}: {sum(c for _, c in parts) / 1e6:.2f}M total", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig_png(fig, 110)


@st.cache_data(show_spinner=False, max_entries=16)
def mc_png(samples, mean, std):
    show = len(samples)
    fig, axes = plt.subplots(1, show + 2, figsize=(2.3 * (show + 2), 2.9))
    for k in range(show):
        axes[k].imshow(samples[k], cmap=EMBER_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[k].set_title(f"pass {k + 1}", fontsize=8.5, loc="left", color=INK2, family="monospace")
    axes[show].imshow(mean, cmap=EMBER_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[show].set_title("mean of all passes", fontsize=8.5, loc="left", color=INK, family="monospace")
    axes[show + 1].imshow(std, cmap=SLATE_CMAP, interpolation="nearest")
    axes[show + 1].set_title("std = uncertainty", fontsize=8.5, loc="left", color=INK, family="monospace")
    style_image_axes(axes)
    fig.tight_layout()
    return fig_png(fig, 110)


@st.cache_data(show_spinner=False)
def browse_table(demo_dir, thr_tuple, dec):
    b = load_bundle()
    thr = RiskThresholds(*thr_tuple)
    rows = []
    for j in range(len(b["y"])):
        y = b["y"][j].astype(np.float32)
        att, mean, std = (b[k][j].astype(np.float32) for k in ("att", "mc_mean", "mc_std"))
        tier = assign_risk_tier(mean, std, thr)
        rows.append({
            "Patch": int(b["orig_idx"][j]),
            "Fire tomorrow %": round(100 * float(y.mean()), 3),
            "F1 attention": patch_scores(att, y, dec)[0],
            "F1 lightweight": patch_scores(mean, y, dec)[0],
            "Max tier": RISK_TIERS[int(tier.max())],
            "High + Extreme %": round(100 * float((tier >= 2).mean()), 3),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- header
bundle, metrics = load_bundle(), load_metrics()

st.markdown("""
<div class="mast">
  <div class="eyebrow">Presidency University &middot; PSAIAC &middot; B.Tech Mini Project 2026</div>
  <h1>Predicting where a wildfire goes next, with a <em>lighter</em> model and an honest error bar.</h1>
  <p>A lightweight linear-attention U-Net forecasts next-day fire spread from satellite and weather data, then turns its
  own uncertainty into a four-level risk triage.</p>
</div>
""", unsafe_allow_html=True)

if bundle is None or metrics is None:
    st.error(f"No demo data found in `{DEMO_DIR}/`. Run `python -m src.export_demo` first "
             "(see DEMO_GUIDE.md), then restart this app.")
    st.stop()

if metrics["meta"].get("synthetic"):
    st.warning("SAMPLE DATA: synthetic maps and placeholder metrics. Not real results.")

attn_m, light_m, load_err = load_models()
live_ok = attn_m is not None

# ------------------------------------------------------------ session state
d = RiskThresholds(**{k: v for k, v in metrics["meta"].get("default_thresholds", {}).items()
                      if k in RiskThresholds.__dataclass_fields__})


def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


THR_DEFAULTS = {"t_mod": _clip(d.prob_moderate, 0.05, 0.9), "t_high": _clip(d.prob_high, 0.05, 0.95),
                "t_ext": _clip(d.prob_extreme, 0.05, 0.99), "t_bump": _clip(d.uncertainty_bump, 0.005, 0.5)}
for k, v in {"pos": 0, "only_fire": True, "mode": MODES[0], "T": 20, "seed": 0, "dec": 0.5, **THR_DEFAULTS}.items():
    st.session_state.setdefault(k, v)
if not live_ok:
    st.session_state["mode"] = MODES[1]

n_total = len(bundle["y"])
fire_frac = bundle["y"].reshape(n_total, -1).mean(1)


def make_pool(only_fire):
    p = np.where(fire_frac > 0)[0] if only_fire else np.arange(n_total)
    if len(p) == 0:
        p = np.arange(n_total)
    return p[np.argsort(-fire_frac[p], kind="stable")]  # biggest fires first


def go_prev():
    st.session_state.pos = max(0, int(st.session_state.pos) - 1)


def go_next():
    st.session_state.pos = min(len(make_pool(st.session_state.only_fire)) - 1, int(st.session_state.pos) + 1)


def go_random():
    st.session_state.pos = int(np.random.randint(len(make_pool(st.session_state.only_fire))))


def resample():
    st.session_state.seed = int(st.session_state.seed) + 1


def reset_thr():
    for k, v in THR_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["dec"] = 0.5


def jump_to(pid):
    hits = np.where(bundle["orig_idx"] == pid)[0]
    if len(hits) == 0:
        return
    j = int(hits[0])
    if fire_frac[j] == 0:
        st.session_state.only_fire = False
    pool_ = make_pool(st.session_state.only_fire)
    st.session_state.pos = int(np.where(pool_ == j)[0][0])
    st.toast(f"Patch #{pid} selected. Open the 'Inspect a patch' tab.")


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.markdown('<div class="brand"><small>Control panel</small>Wildfire risk explorer</div>', unsafe_allow_html=True)

sb.markdown("## Patch")
c1, c2 = sb.columns(2)
c1.button("‹ Prev", on_click=go_prev, key="b_prev")
c2.button("Next ›", on_click=go_next, key="b_next")
sb.button("Random patch", on_click=go_random, key="b_rand")
only_fire = sb.checkbox("Only patches with fire", key="only_fire")
pool = make_pool(only_fire)
st.session_state.pos = int(min(max(int(st.session_state.pos), 0), len(pool) - 1))
if len(pool) > 1:
    sb.slider("Test patch (0 = largest fire)", 0, len(pool) - 1, key="pos")
i = int(pool[st.session_state.pos])
sb.markdown(f'<div class="cap"><b>Patch #{int(bundle["orig_idx"][i])}</b> · {st.session_state.pos + 1} of {len(pool)}<br>'
            f'{100 * fire_frac[i]:.2f}% of pixels burn next day</div>', unsafe_allow_html=True)

sb.markdown("## Inference")
sb.radio("Mode", MODES, key="mode", disabled=not live_ok, horizontal=True)
live = live_ok and st.session_state.mode == MODES[0]
sb.slider("MC-Dropout samples (T)", 5, 50, step=5, key="T", disabled=not live)
sb.button("Resample dropout", on_click=resample, key="b_resample", disabled=not live,
          help="Draws a new set of random dropout masks; shows how stable the uncertainty is.")
if not live_ok:
    sb.caption(f"Live inference unavailable ({load_err or 'PyTorch not installed'}). Showing precomputed outputs.")

sb.markdown("## Decision and risk")
sb.slider("Decision threshold (F1, IoU, error map)", 0.1, 0.9, step=0.05, key="dec")
with sb.expander("Risk tier thresholds", expanded=True):
    st.slider("Moderate ≥ mean prob", 0.05, 0.9, step=0.01, key="t_mod")
    st.slider("High ≥", 0.05, 0.95, step=0.01, key="t_high")
    st.slider("Extreme ≥", 0.05, 0.99, step=0.01, key="t_ext")
    st.slider("Uncertainty bump (std ≥)", 0.005, 0.5, step=0.005, key="t_bump")
    st.button("Reset to calibrated values", on_click=reset_thr, key="b_reset")
    st.caption("Calibrated thresholds were chosen on the validation split.")

s = st.session_state
thr = RiskThresholds(s.t_mod, max(s.t_high, s.t_mod), max(s.t_ext, s.t_high, s.t_mod), s.t_bump)
dec = float(s.dec)

# ------------------------------------------------------------- compute patch
x = bundle["x"][i].astype(np.float32)
y = bundle["y"][i].astype(np.float32)
prev = (x[11] > 0).astype(np.float32)  # PrevFireMask is z-scored per patch; positive = fire

source = "Precomputed"
mc = None
if live:
    try:
        att, att_ms = det_predict(attn_m, x, DEMO_DIR, "attention")
        mc = mc_predict(light_m, x, int(s.T), int(s.seed), DEMO_DIR)
        mean, std = mc["mean"], mc["std"]
        source = f"Live · T={int(s.T)} · seed {int(s.seed)}"
        sb.success(f"Inference: {att_ms + mc['ms']:.0f} ms on CPU (both models)")
    except Exception as e:  # noqa: BLE001
        live, mc = False, None
        sb.warning(f"Live inference failed ({type(e).__name__}); using precomputed outputs.")
if not live:
    att = bundle["att"][i].astype(np.float32)
    mean = bundle["mc_mean"][i].astype(np.float32)
    std = bundle["mc_std"][i].astype(np.float32)

tier = assign_risk_tier(mean, std, thr)
report = risk_tier_report(mean, std, thr)

tab_live, tab_browse, tab_whatif, tab_cmp, tab_hood, tab_about = st.tabs(
    ["Inspect a patch", "Browse all patches", "What-if lab", "Accuracy and cost", "Under the hood", "About"])

# ---------------------------------------------------------------- inspect
with tab_live:
    panels = [("A · Fire today", prev, MASK_CMAP, (0, 1)),
              ("B · Ground truth: fire tomorrow", y, MASK_CMAP, (0, 1)),
              ("C · Attention U-Net (7.85M params)", att, EMBER_CMAP, (0, 1)),
              ("D · Lightweight U-Net (1.45M params)", mean, EMBER_CMAP, (0, 1)),
              ("E · Uncertainty (std of MC passes)", std, SLATE_CMAP, (0, max(float(std.max()), 1e-3))),
              ("F · Risk tier map", tier, TIER_CMAP, (0, 3))]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for ax, (title, arr, cmap, (lo, hi)) in zip(axes.ravel(), panels):
        ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=9.5, loc="left", color=INK2, family="monospace")
    style_image_axes(axes)
    fig.tight_layout()
    map_png = fig_png(fig, 140)
    st.markdown(f'<div class="src">{source}</div>', unsafe_allow_html=True)
    st.image(map_png)

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        mt = report["max_tier"]
        mti = RISK_TIERS.index(mt)
        st.markdown(f'<div class="eyebrow">Highest risk tier &middot; mean confidence {report["mean_confidence"]:.3f}</div>'
                    f'<div class="bigtier"><span class="chip" style="background:{TIER_COLORS[mti]}"></span>{mt}</div>'
                    f'<div class="quote">{report["recommendation"]}</div>', unsafe_allow_html=True)
    with col_b:
        rows = "".join(
            f'<div class="tierrow"><span>{n}</span><div class="t" style="width:{max(report["tier_pixel_pct"][n], 0.5)}%;'
            f'background:{TIER_COLORS[k]}"></div><span style="text-align:right">{report["tier_pixel_pct"][n]:.1f}%</span></div>'
            for k, n in enumerate(RISK_TIERS))
        f1a, ioua = patch_scores(att, y, dec)
        f1l, ioul = patch_scores(mean, y, dec)
        st.markdown(f'<div class="eyebrow" style="border-bottom:1px solid {INK};padding-bottom:.3rem;margin-bottom:.4rem">Share of this patch by tier</div>{rows}'
                    f'<div class="eyebrow" style="border-bottom:1px solid {INK};padding-bottom:.3rem;margin:1rem 0 .4rem">This patch, threshold {dec:.2f}</div>'
                    f'<div class="tierrow" style="grid-template-columns:1fr 4rem 4rem"><span>Attention U-Net</span><span>F1 {fnum(f1a, 2)}</span><span>IoU {fnum(ioua, 2)}</span></div>'
                    f'<div class="tierrow" style="grid-template-columns:1fr 4rem 4rem"><span>Lightweight U-Net</span><span>F1 {fnum(f1l, 2)}</span><span>IoU {fnum(ioul, 2)}</span></div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="cap">Risk tiers combine mean probability with predictive uncertainty: a pixel whose '
                'MC-Dropout std exceeds the bump threshold is escalated one tier. Move the sidebar sliders to see the triage change.</div>',
                unsafe_allow_html=True)

    with st.expander("Where do the models get it right or wrong?"):
        ea, el = error_map(att, y, dec), error_map(mean, y, dec)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, (title, m) in zip(axes, (("Attention U-Net", ea), ("Lightweight U-Net", el))):
            ax.imshow(m, cmap=ERR_CMAP, vmin=0, vmax=3, interpolation="nearest")
            ax.set_title(title, fontsize=9.5, loc="left", color=INK2, family="monospace")
        style_image_axes(axes)
        fig.tight_layout()
        st.image(fig_png(fig, 120))
        st.markdown(legend([("hit (true positive)", ERR_COLORS[1]), ("false alarm", ERR_COLORS[2]),
                            ("missed fire", ERR_COLORS[3])]), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(
            [(n, int((m == 1).sum()), int((m == 2).sum()), int((m == 3).sum()))
             for n, m in (("Attention U-Net", ea), ("Lightweight U-Net", el))],
            columns=["Model", "Hits (pixels)", "False alarms", "Missed fire"]), hide_index=True)
        st.markdown(f'<div class="cap">Pixels classified at decision threshold {dec:.2f} (change it in the sidebar).</div>',
                    unsafe_allow_html=True)

    report_json = json.dumps({
        "patch": int(bundle["orig_idx"][i]), "source": source,
        "thresholds": {"moderate": thr.prob_moderate, "high": thr.prob_high, "extreme": thr.prob_extreme,
                       "uncertainty_bump": thr.uncertainty_bump},
        "decision_threshold": dec,
        "risk_report": report,
        "scores": {"attention": {"F1": fnum(f1a), "IoU": fnum(ioua)}, "lightweight": {"F1": fnum(f1l), "IoU": fnum(ioul)}},
    }, indent=2)
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download figure (PNG)", map_png, file_name=f"patch_{int(bundle['orig_idx'][i])}.png", mime="image/png")
    d2.download_button("Download report (JSON)", report_json, file_name=f"patch_{int(bundle['orig_idx'][i])}_report.json",
                       mime="application/json")

# ------------------------------------------------------------------ browse
with tab_browse:
    st.markdown("### Every patch in the demo bundle")
    st.markdown('<div class="cap">Uses the precomputed predictions, the current risk thresholds and the decision threshold from the sidebar.</div>',
                unsafe_allow_html=True)
    df = browse_table(DEMO_DIR, (thr.prob_moderate, thr.prob_high, thr.prob_extreme, thr.uncertainty_bump), dec)
    f1, f2, f3 = st.columns(3)
    min_fire = f1.number_input("Min fire % tomorrow", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="bf_min")
    sort_by = f2.selectbox("Sort by", ["Fire tomorrow %", "F1 lightweight", "F1 attention", "High + Extreme %", "Patch"], key="bf_sort")
    tier_f = f3.multiselect("Max tier", RISK_TIERS, default=RISK_TIERS, key="bf_tier")
    view = df[(df["Fire tomorrow %"] >= min_fire) & df["Max tier"].isin(tier_f)]
    view = view.sort_values(sort_by, ascending=(sort_by == "Patch"), na_position="last")
    st.markdown(f'<div class="eyebrow">{len(view)} of {len(df)} patches</div>', unsafe_allow_html=True)
    if len(view) == 0:
        st.info("No patches match these filters.")
    else:
        st.dataframe(view, hide_index=True, column_config={
            "Fire tomorrow %": st.column_config.NumberColumn(format="%.2f"),
            "F1 attention": st.column_config.NumberColumn(format="%.3f"),
            "F1 lightweight": st.column_config.NumberColumn(format="%.3f"),
            "High + Extreme %": st.column_config.NumberColumn(format="%.2f")})
        b1, b2, b3 = st.columns([1, 1, 2])
        pid = b1.selectbox("Open patch", view["Patch"].tolist(), key="bf_open")
        b2.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        b2.button("Select it", on_click=jump_to, args=(pid,), key="bf_jump")
        b3.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        b3.download_button("Download table (CSV)", view.to_csv(index=False).encode(), file_name="patches.csv", mime="text/csv")

        has = view.dropna(subset=["F1 attention", "F1 lightweight"])
        has = has[has["Fire tomorrow %"] > 0]
        cA, cB = st.columns(2)
        with cA:
            if len(has):
                fig, ax = plt.subplots(figsize=(5.4, 4))
                ax.scatter(has["F1 attention"], has["F1 lightweight"], s=22, color=EMBER, alpha=0.8, edgecolors="none")
                ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xlabel("F1, Attention U-Net"); ax.set_ylabel("F1, Lightweight U-Net")
                ax.set_title(f"Per-patch F1 ({len(has)} fire patches)", loc="left", fontsize=10)
                st.image(fig_png(fig, 110))
                st.markdown('<div class="cap">Points near the dashed line mean the two models agree patch by patch.</div>',
                            unsafe_allow_html=True)
            else:
                st.info("No fire patches in the current selection.")
        with cB:
            counts = view["Max tier"].value_counts().reindex(RISK_TIERS, fill_value=0)
            fig, ax = plt.subplots(figsize=(5.4, 4))
            ax.bar(RISK_TIERS, counts.values, color=TIER_COLORS)
            ax.set_ylabel("Patches"); ax.grid(axis="x", visible=False)
            ax.set_title("Highest tier reached per patch", loc="left", fontsize=10)
            st.image(fig_png(fig, 110))

# ----------------------------------------------------------------- what-if
with tab_whatif:
    st.markdown("### What if the weather were different?")
    st.markdown('<div class="cap">Shifts the normalised input channels of the selected patch, reruns the lightweight model and '
                'shows how the risk map moves. This is a sensitivity probe on the model, not a weather forecast.</div>',
                unsafe_allow_html=True)
    if not live:
        st.info("The what-if lab needs live inference (PyTorch and the checkpoints in demo/). It is off in precomputed mode.")
    else:
        scen = st.selectbox("Scenario", list(PRESETS) + ["Custom (one channel)"], key="wi_scen")
        if scen in PRESETS:
            edits = {CHANNELS.index(k): (v, 1.0) for k, v in PRESETS[scen].items()}
            st.markdown(f'<div class="cap">Offsets in standard-deviation units: '
                        + ", ".join(f"{k} {v:+.1f}" for k, v in PRESETS[scen].items()) + ".</div>", unsafe_allow_html=True)
        else:
            w1, w2, w3 = st.columns(3)
            ch = w1.selectbox("Channel", CHANNELS[:11], index=2, key="wi_ch")
            shift = w2.slider("Shift (std units)", -3.0, 3.0, 1.5, 0.25, key="wi_shift")
            scale = w3.slider("Scale", 0.0, 2.0, 1.0, 0.1, key="wi_scale")
            edits = {CHANNELS.index(ch): (shift, scale)}
        if st.button("Run scenario", key="wi_run", type="primary"):
            xm = x.copy()
            for c, (sh, sc) in edits.items():
                xm[c] = x[c] * sc + sh
            try:
                base = mc_predict(light_m, x, int(s.T), int(s.seed), DEMO_DIR)
                new = mc_predict(light_m, xm, int(s.T), int(s.seed), DEMO_DIR)
                st.session_state["wi_result"] = {"idx": i, "scen": scen, "T": int(s.T), "seed": int(s.seed),
                                                 "m0": base["mean"], "s0": base["std"], "m1": new["mean"], "s1": new["std"]}
            except Exception as e:  # noqa: BLE001
                st.error(f"Scenario failed ({type(e).__name__}: {e}).")
        res = st.session_state.get("wi_result")
        if res is None:
            st.markdown('<div class="cap">Pick a scenario and press Run scenario.</div>', unsafe_allow_html=True)
        elif res["idx"] != i:
            st.info("The last result was for a different patch. Press Run scenario again for this one.")
        else:
            t0_, t1_ = assign_risk_tier(res["m0"], res["s0"], thr), assign_risk_tier(res["m1"], res["s1"], thr)
            diff = res["m1"] - res["m0"]
            lim = max(float(np.abs(diff).max()), 1e-3)
            fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
            for ax, (ttl, arr, cm, lo, hi) in zip(axes, [
                    ("Original", res["m0"], EMBER_CMAP, 0, 1), (f"Scenario", res["m1"], EMBER_CMAP, 0, 1),
                    ("Change in probability", diff, DIFF_CMAP, -lim, lim), ("Scenario risk tiers", t1_, TIER_CMAP, 0, 3)]):
                ax.imshow(arr, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
                ax.set_title(ttl, fontsize=9, loc="left", color=INK2, family="monospace")
            style_image_axes(axes)
            fig.tight_layout()
            st.image(fig_png(fig, 120))
            n0, n1 = int((res["m0"] >= dec).sum()), int((res["m1"] >= dec).sum())
            h0, h1 = 100 * float((t0_ >= 2).mean()), 100 * float((t1_ >= 2).mean())
            m1_, m2_, m3_ = st.columns(3)
            m1_.metric(f"Pixels ≥ {dec:.2f}", n1, delta=n1 - n0)
            m2_.metric("High + Extreme share", f"{h1:.2f}%", delta=f"{h1 - h0:+.2f} pts")
            m3_.metric("Highest tier", RISK_TIERS[int(t1_.max())],
                       delta=None if t1_.max() == t0_.max() else f"was {RISK_TIERS[int(t0_.max())]}", delta_color="off")
            st.markdown(f'<div class="cap">Scenario: {res["scen"]}. Both runs use the same dropout seed ({res["seed"]}), so the '
                        'difference comes from the input change alone.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------- comparison
with tab_cmp:
    A, L, LM = metrics["attention"], metrics["lightweight"], metrics["lightweight_mc"]
    pr = A["efficiency"]["parameters"] / L["efficiency"]["parameters"]
    fr = (A["efficiency"]["FLOPs_G"] / L["efficiency"]["FLOPs_G"]) if A["efficiency"].get("FLOPs_G") and L["efficiency"].get("FLOPs_G") else None
    ca, cl = A.get("efficiency_cpu", {}), L.get("efficiency_cpu", {})
    cpu = ca["latency_ms_per_image"] / cl["latency_ms_per_image"] if ca.get("latency_ms_per_image") and cl.get("latency_ms_per_image") else None
    glance([
        (f"{pr:.1f}", "&times;", "fewer parameters", f"{A['efficiency']['parameters_M']}M down to {L['efficiency']['parameters_M']}M"),
        (f"{fr:.1f}" if fr else "n/a", "&times;" if fr else "", "fewer FLOPs per patch",
         f"{A['efficiency']['FLOPs_G']}G down to {L['efficiency']['FLOPs_G']}G" if fr else ""),
        (fnum(L["accuracy"]["AUC-PR"]), "", "lightweight AUC-PR", f"Attention U-Net: {fnum(A['accuracy']['AUC-PR'])}"),
        ((f"{cpu:.1f}", "&times;", "faster on CPU", f"{ca['latency_ms_per_image']:.1f} ms down to {cl['latency_ms_per_image']:.1f} ms") if cpu
         else (fnum(L["accuracy"]["F1"]), "", "lightweight F1", f"Attention U-Net: {fnum(A['accuracy']['F1'])}")),
    ])

    rows = [("AUC-PR", A["accuracy"]["AUC-PR"], L["accuracy"]["AUC-PR"]),
            ("F1", A["accuracy"]["F1"], L["accuracy"]["F1"]),
            ("IoU", A["accuracy"]["IoU"], L["accuracy"]["IoU"]),
            ("Parameters (M)", A["efficiency"]["parameters_M"], L["efficiency"]["parameters_M"]),
            ("FLOPs (G)", A["efficiency"]["FLOPs_G"], L["efficiency"]["FLOPs_G"]),
            (f"Latency, {A['efficiency']['device']} (ms)", A["efficiency"]["latency_ms_per_image"],
             L["efficiency"]["latency_ms_per_image"])]
    if ca.get("latency_ms_per_image") and cl.get("latency_ms_per_image"):
        rows.append(("Latency, CPU (ms)", ca["latency_ms_per_image"], cl["latency_ms_per_image"]))
    cmp_df = pd.DataFrame(rows, columns=["Metric", "Attention U-Net", "Lightweight U-Net"])
    st.dataframe(cmp_df, hide_index=True)
    st.markdown(f'<div class="cap">Held-out NDWS test split. Lightweight AUC-PR using the MC-Dropout mean: {fnum(LM["accuracy"]["AUC-PR"])}.</div>',
                unsafe_allow_html=True)
    e1, e2, _ = st.columns([1, 1, 2])
    e1.download_button("Download table (CSV)", cmp_df.to_csv(index=False).encode(), file_name="model_comparison.csv", mime="text/csv")
    e2.download_button("Download all metrics (JSON)", json.dumps(metrics, indent=2), file_name="metrics.json", mime="application/json")

    cA, cB = st.columns(2)
    cA.image(chart_png("pr", DEMO_DIR))
    cB.image(chart_png("f1", DEMO_DIR))

    st.markdown("### Can the uncertainty be trusted?")
    cal = pd.DataFrame(metrics["risk_calibration"])
    cal["observed_fire_rate"] = cal["observed_fire_rate"].map(lambda v: "n/a" if v is None else f"{100 * v:.1f}%")
    cal["pct_of_pixels"] = cal["pct_of_pixels"].map(lambda v: f"{v:.2f}%")
    cal = cal[["tier", "pct_of_pixels", "observed_fire_rate"]]
    cal.columns = ["Tier", "% of all pixels", "Observed fire rate"]
    st.dataframe(cal, hide_index=True)
    u = metrics["uncertainty_stats"]
    st.markdown(f'<div class="cap">A sound triage shows the observed fire rate climbing with the tier. Mean MC-Dropout std is '
                f'{u["mean_std_correct_pixels"]:.4f} on correctly classified pixels and {u["mean_std_wrong_pixels"]:.4f} on '
                f'misclassified ones; the uncertainty bump escalated {100 * u["fraction_pixels_bumped"]:.2f}% of pixels.</div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------- under the hood
with tab_hood:
    st.markdown("### What actually happens to one patch")
    st.markdown(f'<div class="cap">Everything below is computed for the patch selected in the sidebar '
                f'(patch #{int(bundle["orig_idx"][i])}).</div>', unsafe_allow_html=True)

    st.markdown("#### 1 · The input: twelve 64 × 64 maps")
    st.image(inputs_png(x))
    st.markdown('<div class="cap">Tensor shape fed to the network: <b>(1, 12, 64, 64)</b>. Each channel was clipped to a physical '
                'range and z-score normalised, which is why the maps look rescaled.</div>', unsafe_allow_html=True)

    if not live:
        st.info("Live model internals need the checkpoints in demo/, PyTorch, and Live mode in the sidebar. "
                "Sections 2 to 4 appear once they are available.")
    else:
        try:
            st.markdown("#### 2 · Inside the lightweight U-Net: shape of the data at every stage")
            rows_ = stage_shapes(light_m, x, DEMO_DIR)
            if rows_:
                st.dataframe(pd.DataFrame(
                    [(n, STAGE_ROLE.get(n, ""), str(s_), f"{t:.1f}") for n, s_, t in rows_],
                    columns=["Stage", "Role", "Output shape (B, C, H, W)", "Cumulative ms"]), hide_index=True)
                st.markdown('<div class="cap">The encoder halves the map size and widens the channels to find patterns; the decoder '
                            'expands back to 64 × 64, reusing early detail through skip connections. The final sigmoid gives a '
                            'fire probability per pixel.</div>', unsafe_allow_html=True)

            st.markdown("#### 3 · Where the parameters live")
            st.image(params_png(attn_m, light_m, DEMO_DIR))

            st.markdown("#### 4 · Where the uncertainty comes from: repeated passes with dropout left on")
            st.image(mc_png(mc["samples"], mc["mean"], mc["std"]))
            st.markdown(f'<div class="cap">Each pass randomly drops different units, so the maps differ slightly. Their average is the '
                        f'prediction; where they disagree the standard deviation is high. {int(s.T)} passes took {mc["ms"]:.0f} ms on CPU '
                        f'(first computation; later views are cached).</div>', unsafe_allow_html=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not display model internals ({type(e).__name__}: {e}).")

    st.markdown("#### 5 · From probability and uncertainty to a risk tier")
    st.code(f"""tier = 0                                   # Low
tier[mean >= {thr.prob_moderate}] = 1                  # Moderate
tier[mean >= {thr.prob_high}] = 2                  # High
tier[mean >= {thr.prob_extreme}] = 3                  # Extreme
tier[std  >= {thr.uncertainty_bump}] += 1   # uncertain pixels move up one tier (max Extreme)""", language="python")

    st.markdown("#### 6 · Project structure")
    st.code("""src/data.py                       read NDWS TFRecords, clip + normalise, PyTorch DataLoaders
src/models/attention_unet.py      baseline Attention U-Net (7.85M parameters)
src/models/lightweight_unet.py    depthwise-separable convs + linear attention (1.45M)
src/training/train.py             focal loss, Adam, early stopping
src/evaluation/evaluate.py        AUC-PR, F1, IoU, parameters, FLOPs, latency
src/uncertainty.py                MC-Dropout sampling, risk tiers, recommendations
src/calibrate_thresholds.py       picks tier thresholds on the validation split
src/export_demo.py                writes metrics.json + demo bundle for this app
app.py / tools/build_dashboard.py this app and the offline dashboard""", language="text")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(f"""
**Problem.** Given twelve remote-sensing channels (terrain, weather, vegetation, drought, population and today's fire mask),
predict which pixels of a 64 × 64 km window will burn tomorrow (NDWS dataset, Huot et al., 2022).

**Models.** An Attention U-Net baseline (7.85 M parameters) and a lightweight U-Net that uses depthwise-separable
convolutions and a single linear-attention block at the bottleneck (1.45 M parameters).

**Decision support.** Monte-Carlo Dropout gives a per-pixel mean and standard deviation. Both feed a four-tier risk
classifier (Low, Moderate, High, Extreme) with an uncertainty-bump rule and a plain-language recommendation. Thresholds
are calibrated on a validation split, never on the test split.

**Limitations.** One dataset (contiguous US). Absolute scores are modest and vary between training runs, so the two
models are best described as comparable in accuracy. FLOPs are counted with `thop` (convolutions, linear layers,
batch-norm). Risk thresholds are not calibrated against real operational outcomes. This is decision support, not an
autonomous system.

*Test split: {metrics['meta']['n_test_patches']} patches · fire pixels: {100 * metrics['meta']['fire_pixel_fraction']:.2f}% · MC samples used for evaluation: {metrics['meta']['mc_samples']}.*
""")
    with st.expander("How to use this app"):
        st.markdown("""
1. **Inspect a patch:** step through test patches with Prev, Next, Random or the slider; the six maps, risk tier and recommendation update.
2. **Sidebar thresholds:** change the risk thresholds and see the tier map move; the decision threshold drives F1, IoU and the error map.
3. **Resample dropout:** draws new dropout masks to show how stable the uncertainty is.
4. **Browse all patches:** filter and sort every patch, download the table, or select one to inspect.
5. **What-if lab:** apply a weather scenario to the current patch and compare the risk before and after.
6. **Under the hood:** the real tensors, layer shapes and dropout passes behind the prediction.
""")
