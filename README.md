# CRISP-AX

Code accompanying the data descriptor:

> **CRISP-AX, a window-level paired 802.11ax CSI–RGB dataset for home activity and
> unoccupied-environment states**

CRISP-AX (**C**SI–**R**GB **I**ndoor **S**ensing **P**airs; AX = IEEE 802.11ax) is a dataset of
7,024 window-level paired samples of WiFi channel state information (CSI) and RGB video,
recorded in a living room, bedroom, bathroom and kitchen. 6,960 windows cover 14 human action
classes from ten subjects; 64 unoccupied windows span 13 annotated environment–condition
combinations.

This repository holds the **acquisition, validation, baseline and figure-generation code**.
It does **not** contain the dataset itself — see [Data availability](#data-availability).

## Repository layout

| Path | Contents | Paper section |
|---|---|---|
| `Params/` | Acquisition commands (FeitCSI injection, video capture, `collect_csi_formal.sh`) and recorded room geometry | Methods 2.1–2.2 |
| `quality_check/` | Release integrity, CSI parsing and continuity checks; unoccupied-condition analysis; figure generation | Technical Validation 4.1–4.3, 4.5 |
| `baseline-csi-2dcnn/` | CSI-only baseline training and evaluation | Technical Validation 4.4 |
| `baseline-video-r3d18/` | Video-only baseline training and evaluation | Technical Validation 4.4 |
| `scripts/autodl/` | Multi-architecture run drivers, weight prefetch, result verification | Technical Validation 4.4 |
| `tools/` | `check_csi_dims.py` (CSI dimension check), `extract_device_state_frames.py` (unoccupied-state video frames) | Methods, Data Records |

## Acquisition

CSI is injected and measured with [FeitCSI](https://feitcsi.kuskosoft.com) on Intel AX210
adapters, on a 5180 MHz 80 MHz channel using the HE-SU frame format, MCS 0, one spatial
stream, one transmit and two receive antennas, at a 10,000 µs injection interval
(nominally ~100 Hz, 996 subcarriers per packet). RGB video is captured at 640×360, 30 fps.
Exact commands are in `Params/Params.md`; the receiver-side script is
`Params/collect_csi_formal.sh`.

## Requirements

Analysis and figure code (`quality_check/requirements-unoccupied-validation.txt`):

```
numpy>=1.26
scipy>=1.11
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
```

The baselines additionally require PyTorch and Torchvision with CUDA.

## Environment used for the reported results

All 105 baseline runs reported in the paper were executed on a single stack:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| NumPy | 2.5.0 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4090 |

Two independent executions with the same GPU model, software stack and random seed produced
bitwise-identical results. Reproduction on different hardware is expected to fall within the
seed-to-seed and fold-to-fold spread reported in the paper.

## Data availability

The dataset is released separately from this repository.
<!-- TODO: add repository name, DOI and access conditions once the data deposit is finalised. -->

## Citation

<!-- TODO: add the article citation and the dataset DOI once assigned. -->

## License

Code in this repository is released under the MIT License (see `LICENSE`).
The dataset is distributed under its own licence, stated with the data deposit.
