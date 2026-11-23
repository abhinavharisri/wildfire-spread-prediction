# Final-review demo kit

Two deliverables, both driven by one export step:

| What | File | Needs on review day |
|---|---|---|
| Live app (real inference, sliders, risk maps) | `app.py` (Streamlit) | laptop + `demo/` folder, no GPU, no dataset |
| Results dashboard (charts + patch explorer) | `dashboard.html` | any browser, offline, double-click |

## 1. Train (Kaggle GPU, ~20 min)
New notebook -> Accelerator: GPU, Internet: On, Add Input: `next-day-wildfire-spread` (fantineh).
```python
!git clone https://github.com/abhinavharisri/wildfire-spread-prediction.git
%cd wildfire-spread-prediction
!pip install -q thop
!find /kaggle/input -name "*.tfrecord" | head -3       # note the folder -> DATA_DIR
!mkdir -p logs
!python -m src.training.train --model attention   --data_dir DATA_DIR --epochs 30 --patience 6 --batch_size 32 2>&1 | tee logs/attention.log
!python -m src.training.train --model lightweight --data_dir DATA_DIR --epochs 30 --patience 6 --batch_size 32 2>&1 | tee logs/lightweight.log
```
## 2. Export everything the demo needs (same notebook, ~3 min)
```python
!python -m src.export_demo --data_dir DATA_DIR --attn_ckpt checkpoints/attention_best.pt --light_ckpt checkpoints/lightweight_best.pt
!zip -qr demo_out.zip demo logs
```
Download `demo_out.zip` from the notebook's Output panel (and back it up to Drive). The printed summary is your **new final numbers**.

## 3. Laptop
```bash
unzip demo_out.zip                       # inside the repo folder -> demo/ and logs/
pip install -r requirements-demo.txt
streamlit run app.py                     # live demo
python tools/build_dashboard.py --demo demo --logs logs --out dashboard.html
```
`dashboard.html` is a single file: copy it to a USB stick / email it to yourself as the no-risk fallback.

## 4. Review-day flow (about 5 minutes)
1. **Dashboard, headline cards** (30 s): "5.4x fewer parameters, 6.3x fewer FLOPs; accuracy comparable or better." Say the CPU-latency number only if it appears in the cards.
2. **Accuracy vs efficiency + PR curve** (45 s): reported together; this is the gap in the literature.
3. **Live app** (2 min): pick a fire patch -> show today / ground truth / both models / uncertainty / risk map. Drag the Moderate threshold and the uncertainty bump and show the map and recommendation change.
4. **Calibration table + training curves** (45 s): observed fire rate should rise with the tier; show the attention model overfitting while the lightweight one does not (only if the curves show it).
5. **Limitations slide** (30 s): single dataset, modest absolute scores, run-to-run variance, uncalibrated thresholds, decision support only.

## 5. Questions to expect
* *Why are scores modest?* NDWS is hard (about 1% fire pixels, ~15k training patches); threshold sweep and regularization checks show it is a capacity ceiling, not a bug.
* *Which model is better?* Say what the table in front of you shows. Earlier runs had the attention model ahead in one run and behind in another; with this few runs claim "comparable accuracy at a fraction of the cost", not "more accurate".
* *Is the uncertainty meaningful?* Point to the calibration table and the correct-vs-wrong-pixel std. If it is flat, say so honestly.
* *Why is GPU latency similar?* 64x64 inputs on a GPU are launch-overhead bound; the CPU latency row is the fair edge-deployment comparison.

## 6. If something breaks
* Streamlit will not start: open `dashboard.html` (needs nothing).
* Live mode greyed out: checkpoints missing from `demo/`; the app falls back to precomputed outputs automatically.
* Never present `dashboard_SAMPLE_PREVIEW.html` or `demo_sample/` (synthetic).
