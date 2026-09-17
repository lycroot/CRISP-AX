# CRISP-AX

Code accompanying the CRISP-AX Data Descriptor.

CRISP-AX (**C**SI–**R**GB **I**ndoor **S**ensing **P**airs; AX = IEEE 802.11ax) provides 7,024 paired
windows of Wi-Fi channel state information (CSI) and RGB video collected in living room, bedroom,
bathroom and kitchen environments. 6,960 windows cover 14 human activity classes performed by
10 participants; the remaining 64 unoccupied windows span 13 annotated environment–condition
combinations. CSI was collected in the 5 GHz band with an 80 MHz bandwidth (996 subcarriers per
packet, nominal sampling rate 100 Hz); RGB video has a resolution of 640 × 360 pixels.

This repository contains the scripts used to process the raw recordings into the released windows
and the code of the single-modality baselines reported in the Technical Validation. The dataset
itself is distributed separately (see [Data availability](#data-availability)).

## Repository layout

| Path | Contents | Paper section |
|---|---|---|
| `data_processing/` | Synchronization, slicing, quality screening, face blurring and release assembly | Methods |
| `baseline-csi-2dcnn/` | CSI-only baselines: `cnn2d`, `resnet18`, `bilstm`, `transformer` | Technical Validation |
| `baseline-video-r3d18/` | Video-only baselines: `r3d_18`, `mc3_18`, `r2plus1d_18` | Technical Validation |

The project was developed under the working name AXHome-MM. The baseline directory names and the
Python package names `axhome_csi` and `axhome_video` are kept from that stage so that the recorded
run configurations remain valid.

## Data processing

`data_processing/` contains the scripts that produced the released windows from the raw recordings.
They read the raw CSI and video recordings and intermediate directories of the project workspace,
which are not distributed, so they document the processing rather than run on the release itself.
The scripts are unchanged from the project workspace except that the package name in import paths
and help texts was changed from `quality_check` to `data_processing`.

| Step | Script | Purpose |
|---:|---|---|
| 1 | `build_csi_video_sync_index.py` | Builds the CSI–video synchronization index. Action start and end times come from the collection manifest; CSI packet times come from the per-packet receiver log, with a linear packet-index mapping when the log and `.dat` packet counts differ; the video frame range comes from the frame-timestamp CSV. |
| 2 | `slice_csi_video_windows.py` | Cuts each action window into a CSI `.dat` file, an MP4 clip and a metadata JSON file, and writes `slice_index.csv`. |
| 3 | `check_csi_slice_quality.py` | Checks the sliced CSI windows against the slice and synchronization indexes (packet counts, packet rate, payload statistics) and the manual video review notes. |
| 4 | `build_problem_sample_cleanup_index.py` | Builds the plan for removing windows that did not pass the checks. |
| 5 | `apply_problem_sample_cleanup.py` | Applies the plan; dry run by default, `--apply` moves the files out of the clean set and rewrites the slice index. |
| 6 | `run_deepmosaics_face_anonymization.py` | Runs DeepMosaics in `add` mode with its face model to blur faces in the clean clips. |
| 7 | `build_final_anonymized_archive.py` | Combines the face-blurred videos with the sliced CSI and metadata into a by-session archive. |
| 8 | `build_dataset_release.py` | Builds the release directory, manifests and release report from the archive. |

Typical invocation from the workspace root:

```bash
WORKSPACE=/path/to/workspace   # contains CSI-Formal/, Video/, quality_report_deep/, ...

python data_processing/build_csi_video_sync_index.py --project-root "$WORKSPACE"
python data_processing/slice_csi_video_windows.py --project-root "$WORKSPACE" \
  --full --group-by-session --output-dir sliced_windows_by_session
python data_processing/check_csi_slice_quality.py --project-root "$WORKSPACE"
python data_processing/build_problem_sample_cleanup_index.py --project-root "$WORKSPACE"
python data_processing/apply_problem_sample_cleanup.py --project-root "$WORKSPACE" --apply
python data_processing/run_deepmosaics_face_anonymization.py --project-root "$WORKSPACE" \
  --deepmosaics-root /path/to/DeepMosaics --apply
python data_processing/build_final_anonymized_archive.py --project-root "$WORKSPACE" \
  --s00-placement canonical_person_environment \
  --output-dir sliced_windows_final_anonymized_canonical_session_20260705
python data_processing/build_dataset_release.py --project-root "$WORKSPACE"
```

Notes:

- Raw inputs are the collection manifest (`CSI-Formal/01_manifest/collection_manifest_raw.csv`), the
  CSI `.dat` files and receiver logs (`CSI-Formal/00_raw/`), and the video recordings with their
  frame-timestamp CSV files (`Video/`). Steps 3 and 4 also read project-internal manual review notes
  and subject/session tables.
- DeepMosaics and its face model are not included in this repository.
- The 46 supplemental S00 windows were synchronized, sliced and face-blurred separately and merged
  in step 7, which places them under their canonical sessions.
- After the release was built, the 65 S01 videos were re-encoded for playback compatibility with
  `tools/transcode_s01_video_compatibility.py` (distributed with the dataset), and one window was
  excluded because its CSI and video were not temporally aligned (`manifests/excluded_samples.csv`).
  Both revisions are described in the release report.
- `build_dataset_release.py` still uses the working name `AXHome-MM-v1` as its default dataset name;
  the release was renamed to CRISP-AX afterwards (see the release report).

## Dataset layout expected by the code

The baseline scripts take `--dataset-root`, which points to the extracted release directory:

```text
CRISP-AX/
├── DATASET_VERSION.txt
├── data/
│   └── Sxx/
│       ├── csi/<action>/<sample_id>.dat
│       ├── metadata/<action>/<sample_id>.json
│       └── video/<action>/<sample_id>.mp4
├── manifests/archive_index.csv
└── reports/
```

## Requirements

- Python 3.10 or later.
- `data_processing/`: NumPy, OpenCV (`cv2`) and FFmpeg; step 6 additionally needs a DeepMosaics
  installation.
- Baselines: `baseline-csi-2dcnn/requirements.txt` and `baseline-video-r3d18/requirements.txt`
  (PyTorch with CUDA is recommended for training; the Video-only baselines also need PyAV).

## Reproducing the baselines

### Single-modality baselines

The reported results comprise seven architectures × two protocols: in-domain (subject × action
stratified 65%/17.5%/17.5% split, seeds 2026–2030) and subject-wise leave-one-subject-out (LOSO,
10 folds). All architectures within a modality share the same input, split and training budget;
no per-architecture hyperparameter search was performed. The Transformer uses a learning rate of
3e-4 and dropout of 0.1. The complete matrix (105 runs) is:

```bash
DATASET_ROOT=/path/to/CRISP-AX
RESULTS=/path/to/results
SEEDS="2026 2027 2028 2029 2030"
PEOPLE="P01 P02 P03 P04 P05 P06 P07 P08 P09 P10"

# Preprocessing caches (once per modality)
(cd baseline-csi-2dcnn && python build_cache.py \
  --dataset-root "$DATASET_ROOT" --cache-dir "$RESULTS/csi-cache" --workers 8)
(cd baseline-video-r3d18 && python build_cache.py \
  --dataset-root "$DATASET_ROOT" --cache-dir "$RESULTS/video-cache" --workers 8)

# CSI-only: 4 architectures x 2 protocols (60 runs)
cd baseline-csi-2dcnn
for spec in "cnn2d 1e-3 0.5" "resnet18 1e-3 0.5" "bilstm 1e-3 0.5" "transformer 3e-4 0.1"; do
  set -- $spec
  python run_in_domain.py --dataset-root "$DATASET_ROOT" --cache-dir "$RESULTS/csi-cache" \
    --output-dir "$RESULTS/in_domain_$1" --arch "$1" --seeds $SEEDS \
    --epochs 50 --batch-size 32 --learning-rate "$2" --weight-decay 0.0 --dropout "$3" \
    --patience 10 --num-workers 8 --device cuda
  python run_loso.py --dataset-root "$DATASET_ROOT" --cache-dir "$RESULTS/csi-cache" \
    --output-dir "$RESULTS/loso_$1" --arch "$1" --test-people $PEOPLE \
    --epochs 50 --batch-size 32 --learning-rate "$2" --weight-decay 0.0 --dropout "$3" \
    --patience 10 --num-workers 8 --device cuda
done
cd ..

# Video-only: 3 architectures x 2 protocols (45 runs).
# In-domain runs reuse the per-sample split of the CSI cnn2d in-domain run.
cd baseline-video-r3d18
for arch in r3d_18 mc3_18 r2plus1d_18; do
  python run_in_domain.py --dataset-root "$DATASET_ROOT" --cache-dir "$RESULTS/video-cache" \
    --reference-assignment-root "$RESULTS/in_domain_cnn2d" \
    --output-dir "$RESULTS/video_in_domain_$arch" --arch "$arch" --seeds $SEEDS \
    --epochs 50 --batch-size 16 --learning-rate 1e-4 --weight-decay 1e-4 \
    --patience 10 --num-workers 8 --device cuda --amp --pretrained
  python run_loso.py --dataset-root "$DATASET_ROOT" --cache-dir "$RESULTS/video-cache" \
    --output-dir "$RESULTS/video_loso_$arch" --arch "$arch" --test-people $PEOPLE \
    --epochs 50 --batch-size 16 --learning-rate 1e-4 --weight-decay 1e-4 \
    --patience 10 --num-workers 8 --device cuda --amp --pretrained
done
cd ..
```

Details of preprocessing, splits and output files are in
[`baseline-csi-2dcnn/README.md`](baseline-csi-2dcnn/README.md) and
[`baseline-video-r3d18/README.md`](baseline-video-r3d18/README.md).

## Environment used for the reported results

All 105 baseline runs reported in the paper were executed on a single software and hardware stack:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| NumPy | 2.5.0 |
| CUDA | 12.8 |
| cuDNN | 9.10.2 |
| GPU | NVIDIA GeForce RTX 4090 |

Every run writes its runtime (library versions, CUDA/cuDNN and GPU model) to `run_config.json`.
Training sets `torch.backends.cudnn.deterministic = True`; two independent executions on the same GPU
model, software stack and seeds produced bitwise-identical metrics for the 75 runs they had in
common. Runs on different hardware are expected to fall within the seed-to-seed and fold-to-fold
spread reported in the paper.

The Video-only baselines use the official Torchvision Kinetics-400 checkpoints:

| Architecture | File | SHA-256 |
|---|---|---|
| `r3d_18` | `r3d_18-b3b3357e.pth` | `b3b3357ead25631ec9c57362ff2128a92d0427e01e2cd184951a44380c3f2e9d` |
| `mc3_18` | `mc3_18-a90a0ba3.pth` | `a90a0ba35ca1242d15b77511ff28bfb29cc596988b5ea36081042f8e2f54212b` |
| `r2plus1d_18` | `r2plus1d_18-91a641e6.pth` | `91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8` |

## Data availability

The dataset is released separately from this repository.
<!-- TODO: add repository name, DOI and access conditions once the data deposit is finalised. -->

## Citation

<!-- TODO: add the article citation and the dataset DOI once assigned. -->

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The dataset is distributed under its own licence, stated with the data deposit.
