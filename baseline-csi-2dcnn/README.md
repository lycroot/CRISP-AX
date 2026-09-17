# CRISP-AX CSI-only baselines

Technical-validation baselines for the CSI modality of CRISP-AX. The code reads
`manifests/archive_index.csv` and the FeitCSI `.dat` files of the extracted release and performs
14-class human activity recognition under two protocols: in-domain learnability and generalization
to unseen subjects.

## Summary

- **Input:** CSI amplitude time–subcarrier maps of the two receive chains, shape `(2, 256, 128)`.
- **Architectures** (selected with `--arch`):
  - `cnn2d` (default): lightweight three-block 2D CNN (32, 64 and 128 channels; dense layers of 256
    and 128 units);
  - `resnet18`: ResNet-18 (BasicBlock, [2, 2, 2, 2]) trained from scratch;
  - `bilstm`: linear packet projection followed by a two-layer bidirectional LSTM;
  - `transformer`: linear packet projection with learned positions and a Transformer encoder.

  All architectures use the same frozen input, splits and training budget. They are reported
  together as evidence that separability does not depend on a single architecture, not as a model
  ranking.
- **Task:** 14 human activity classes. The 64 unoccupied `background_idle` windows have no subject
  identity and are not used.
- **Protocol A (in-domain):** stratified by `(person_id, action_id)` into 65%/17.5%/17.5%
  train/validation/test, repeated for seeds 2026–2030. Each seed yields 4,547 / 1,192 / 1,221
  samples (6,960 in total); seeds change the per-sample assignment, not the set sizes.
- **Protocol B (unseen subjects):** 10-fold subject-wise LOSO. Each fold uses one test subject, one
  validation subject and the remaining eight subjects for training.
- **Class imbalance:** balanced class weights computed from the current training split only.
- **Metrics:** Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, per-class F1 and confusion
  matrices.

Both protocols share preprocessing, models, class weighting, early stopping and metrics. The
`cnn2d` design is inspired by the three-layer 2D CNN of the EHUNAM Data Descriptor but uses
amplitude only (see [REFERENCES.md](REFERENCES.md)).

## Directory

```text
baseline-csi-2dcnn/
├── axhome_csi/          # parsing, preprocessing, splits, models, training, evaluation
├── tests/               # parser, split, cache, metric and training smoke tests
├── build_cache.py       # optional: precompute amplitude maps
├── inspect_dataset.py   # summarize the release manifest
├── train_in_domain.py   # one in-domain seed
├── run_in_domain.py     # several in-domain seeds plus summary
├── train_fold.py        # one LOSO fold
├── run_loso.py          # all LOSO folds plus summary
├── REFERENCES.md
└── requirements.txt
```

## Usage

### 1. Install

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Python 3.10–3.12 and PyTorch 2.2 or later are recommended. If a CUDA build of PyTorch is already
installed, install only NumPy and Matplotlib to keep it. `requirements.txt` supports NumPy 1.26 and
2.x.

### 2. Inspect the release

```bash
python inspect_dataset.py --dataset-root /path/to/CRISP-AX --validate-paths --output profile.json
```

The release should report 7,024 samples: 6,960 human activity samples and 64 unoccupied samples.

### 3. Build the preprocessing cache

The cache is optional but recommended, because all runs read the same data repeatedly.

```bash
python build_cache.py --dataset-root /path/to/CRISP-AX --cache-dir /path/to/csi-cache --workers 8
```

The cache holds 6,960 `float32` arrays of shape `(2, 256, 128)` (about 1.70 GiB uncompressed). The
script checks that the number of parsed packets matches the manifest; failures are written to
`cache_build_report.json` and produce a non-zero exit code. Use `--limit 20` for a quick check.

### 4. Protocol A: in-domain, five seeds

```bash
python -u run_in_domain.py \
  --dataset-root /path/to/CRISP-AX \
  --cache-dir /path/to/csi-cache \
  --output-dir results/in_domain_cnn2d \
  --arch cnn2d \
  --seeds 2026 2027 2028 2029 2030 \
  --epochs 50 --batch-size 32 --num-workers 8 --device cuda
```

A single seed can be run with `train_in_domain.py` and `--seed`.

### 5. Protocol B: LOSO

```bash
python -u run_loso.py \
  --dataset-root /path/to/CRISP-AX \
  --cache-dir /path/to/csi-cache \
  --output-dir results/loso_cnn2d \
  --arch cnn2d \
  --epochs 50 --batch-size 32 --num-workers 8 --device cuda
```

`--test-people P01 P02` runs a subset of folds, and `train_fold.py --test-person P01 --val-person P02`
runs a single fold. For debugging, `--epochs 1 --batch-size 8 --num-workers 2` is sufficient. If GPU
memory is insufficient, reduce `--batch-size` rather than changing the protocol.

The full seven-architecture command matrix used for the paper is given in the repository
[README](../README.md#single-modality-baselines).

## LOSO folds

| Fold | Test subject | Validation subject | Training subjects |
|---:|---|---|---:|
| 1 | P01 | P02 | 8 |
| 2 | P02 | P03 | 8 |
| 3 | P03 | P04 | 8 |
| 4 | P04 | P05 | 8 |
| 5 | P05 | P06 | 8 |
| 6 | P06 | P07 | 8 |
| 7 | P07 | P08 | 8 |
| 8 | P08 | P09 | 8 |
| 9 | P09 | P10 | 8 |
| 10 | P10 | P01 | 8 |

All environments, sessions and trials of a subject stay in the same split.

## Preprocessing

1. Parse records using the FeitCSI 272-byte header.
2. Rebuild complex CSI from signed `int16` real and imaginary parts, giving
   `(packet, RX, TX, subcarrier)`.
3. Check that each file's packet count equals `csi_packets_written`.
4. Compute `log1p(abs(CSI))`; uncalibrated phase is not used.
5. Identify zero or very low-energy subcarriers from the per-sample median subcarrier energy and
   repair them by linear interpolation along the fixed frequency axis.
6. Select 128 evenly spaced positions on the fixed 996-subcarrier axis and 256 evenly spaced packets
   from the window, so that the j-th frequency position corresponds across samples.
7. Z-score each RX channel within the sample.

Released samples are already synchronized activity windows, so no re-slicing is performed.

## Outputs

Each fold or seed directory contains:

- `best_model.pt`: checkpoint with the best validation Macro-F1;
- `run_config.json`: hyperparameters, classes, device, runtime and subject split;
- `history.csv`: per-epoch training and validation loss and metrics;
- `metrics.json`: overall and per-class test metrics;
- `classification_report.csv`: per-class precision, recall, F1 and support;
- `predictions.csv`: per-sample true and predicted labels;
- `confusion_matrix_counts.csv`, `confusion_matrix_counts.png`, `confusion_matrix_normalized.png`.

In-domain seed directories also contain `split_assignments.csv` (per-sample split and seed). The
multi-seed root contains `in_domain_runs.csv`, `in_domain_summary.json` (mean, population standard
deviation, minimum and maximum of each metric) and the row-normalized mean confusion matrix
(`confusion_matrix_mean_normalized.csv` / `.png`). A full LOSO run writes `loso_folds.csv` and
`loso_summary.json`.

`complete` in a summary is `true` only if every requested seed or fold succeeded. A failed run is
recorded with its error, the other runs are kept, and the process exits with a non-zero code.

## Tests

```bash
python -m unittest discover -s tests -v
```

Data-layer tests run without PyTorch; with PyTorch installed, model forward passes and a one-epoch
synthetic training run are also tested.
