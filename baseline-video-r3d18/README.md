# CRISP-AX Video-only baselines

Technical-validation baselines for the RGB video modality of CRISP-AX. The models are Torchvision
video backbones pretrained on Kinetics-400 (selected with `--arch` from `r3d_18`, `mc3_18` and
`r2plus1d_18`; default `r3d_18`), fine-tuned for 14-class classification of the 6,960 human activity
videos. The 64 `background_idle` windows with `person_id=NONE` are not used.

The three backbones are variants of the same 18-layer VideoResNet family that differ in how they
factorize spatiotemporal convolutions. They share the same clip input, splits and training budget.

## Fixed settings

- Input: 16 frames sampled uniformly over the full synchronized window.
- Cache: RGB, `uint8`, `(16, 128, 171, 3)`.
- Models: `torchvision.models.video.{r3d_18, mc3_18, r2plus1d_18}` with the corresponding
  `KINETICS400_V1` weights (`R3D_18_Weights`, `MC3_18_Weights`, `R2Plus1D_18_Weights`); all
  parameters are fine-tuned. Training fails instead of silently falling back to random
  initialization if the weights are unavailable.
- Clip input `(3, 16, 112, 112)`: random `112 × 112` crop and horizontal flip (p = 0.5) for training,
  center crop for validation and test.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- At most 50 epochs, early stopping on validation Macro-F1 with patience 10.
- Batch size 16, automatic mixed precision on CUDA by default.
- Class imbalance: balanced class weights from the training split.
- Metrics: Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1.
- Aggregation: mean, population standard deviation (`ddof=0`), minimum and maximum; confusion
  matrices are row-normalized per run and then averaged.

## Protocols

### In-domain, five seeds

Seeds 2026–2030. Samples are stratified by `(person_id, action_id)`, sorted by `sample_id` within each
stratum, shuffled with the seed and split 65%/17.5%/17.5%:

```text
train       4,547
validation  1,192
test        1,221
total       6,960
```

Batch runs require `--reference-assignment-root`, which points to a completed CSI-only in-domain run.
Every `(sample_id, split, seed)` row is checked against that run, so the CSI-only and Video-only
in-domain results use identical splits.

### Subject-wise LOSO, ten folds

Each subject is the test subject once, the next subject is the validation subject and the remaining
eight subjects are used for training (`P01→P02`, `P02→P03`, …, `P10→P01`). Every fold uses seed 2026.
All environments, sessions and trials of a subject stay in the same split.

## Usage

### Install

```bash
python -m pip install -r requirements.txt
python -c "import torch, torchvision, av; print(torch.__version__, torchvision.__version__, av.__version__, torch.cuda.is_available())"
```

If a CUDA build of PyTorch and Torchvision is already installed, keep it and add only PyAV, Pillow,
NumPy and Matplotlib.

### 1. Inspect the manifest

```bash
python -B inspect_dataset.py --dataset-root /path/to/CRISP-AX --validate-paths --output video_profile.json
```

### 2. Build the video cache (once)

```bash
python -u -B build_cache.py --dataset-root /path/to/CRISP-AX --cache-dir /path/to/video-cache --workers 8
```

The cache can be resumed; existing files with a mismatching shape, type or configuration are
rejected. A successful build reports `requested=6960`, `completed=6960`, `failed=[]` and
`complete=true` in `cache_build_report.json`.

### 3. In-domain, five seeds

```bash
python -u -B run_in_domain.py \
  --dataset-root /path/to/CRISP-AX \
  --cache-dir /path/to/video-cache \
  --reference-assignment-root results/in_domain_cnn2d \
  --output-dir results/video_in_domain_r3d_18 \
  --arch r3d_18 \
  --seeds 2026 2027 2028 2029 2030 \
  --epochs 50 --batch-size 16 --num-workers 8 --device cuda --amp
```

### 4. LOSO, ten folds

```bash
python -u -B run_loso.py \
  --dataset-root /path/to/CRISP-AX \
  --cache-dir /path/to/video-cache \
  --output-dir results/video_loso_r3d_18 \
  --arch r3d_18 \
  --epochs 50 --batch-size 16 --num-workers 8 --seed 2026 --device cuda --amp
```

`--test-people P01 P02` runs only the listed folds (for debugging). Single runs are available through
`train_in_domain.py --seed 2026` and `train_fold.py --test-person P01 --val-person P02`.

The full seven-architecture command matrix used for the paper is given in the repository
[README](../README.md#single-modality-baselines).

## Outputs and failure handling

Each seed or fold directory contains the checkpoint, run configuration, split audit table, training
history, per-sample predictions, classification report and confusion matrices. Batch directories
also contain a run table, a summary JSON and the mean normalized confusion matrix.

`run_config.json` records the Python, NumPy, PyTorch, Torchvision, PyAV, Pillow, CUDA/cuDNN and GPU
versions, together with the Kinetics-400 checkpoint file name, source URL and SHA-256.

Batch entry points catch errors of individual seeds or folds, keep the successful runs and refresh
the partial summary after each run. If any run fails, `complete=false` and the process exits with a
non-zero code.

If GPU memory is insufficient, reduce `--batch-size` from 16 to 8 or 4 rather than changing the
16-frame input, the 112-pixel crop or the splits.

## Tests

```bash
python -m unittest discover -s tests -v
```

See [REFERENCES.md](REFERENCES.md) for the model and pretraining references.
