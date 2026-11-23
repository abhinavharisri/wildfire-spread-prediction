"""
Builds a single self-contained, offline HTML dashboard from the demo/ folder.

    python tools/build_dashboard.py --demo demo --logs logs --out dashboard.html

Inputs : demo/metrics.json, demo/demo_bundle.npz   (from `python -m src.export_demo`)
         logs/attention.log, logs/lightweight.log  (optional; training curves)
Output : one .html file - double-click to open, no server, no internet needed.
"""
import argparse
import base64
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCH_RE = re.compile(r"epoch\s+(\d+)/\d+\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def parse_log(path):
    ep, tr, va = [], [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                ep.append(int(m.group(1)))
                tr.append(float(m.group(2)))
                va.append(float(m.group(3)))
    return {"epoch": ep, "train_loss": tr, "val_loss": va} if ep else None


def load_history(demo_dir, logs_dir):
    hist_path = os.path.join(demo_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            return json.load(f)
    if logs_dir:
        a = os.path.join(logs_dir, "attention.log")
        l = os.path.join(logs_dir, "lightweight.log")
        if os.path.exists(a) and os.path.exists(l):
            ha, hl = parse_log(a), parse_log(l)
            if ha and hl:
                return {"attention": ha, "lightweight": hl}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="demo")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args()

    with open(os.path.join(args.demo, "metrics.json")) as f:
        metrics = json.load(f)
    z = np.load(os.path.join(args.demo, "demo_bundle.npz"))

    std_max = float(max(np.clip(z["mc_std"][i].astype(np.float32), 0, 0.5).max() for i in z["featured"]))
    std_max = max(std_max, 1e-3)
    patches = []
    for i in z["featured"]:
        y = z["y"][i].astype(np.uint8)
        patches.append({
            "idx": int(z["orig_idx"][i]),
            "fire": float(y.mean()),
            "prev": b64((z["x"][i][11].astype(np.float32) > 0).astype(np.uint8)),
            "y": b64(y),
            "att": b64(np.round(np.clip(z["att"][i].astype(np.float32), 0, 1) * 255)),
            "mean": b64(np.round(np.clip(z["mc_mean"][i].astype(np.float32), 0, 1) * 255)),
            "std": b64(np.round(np.clip(z["mc_std"][i].astype(np.float32), 0, std_max) / std_max * 255)),
        })

    history = load_history(args.demo, args.logs)
    data = {"metrics": metrics, "patches": patches, "history": history, "std_max": round(std_max, 5)}

    with open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8") as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, {len(patches)} patches, "
          f"history={'yes' if history else 'no'}, synthetic={metrics['meta'].get('synthetic', False)})")


if __name__ == "__main__":
    main()
