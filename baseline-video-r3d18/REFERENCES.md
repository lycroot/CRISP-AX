# References for the Video-only baselines

## Spatiotemporal convolution variants (R3D-18, MC3-18, R(2+1)D-18)

Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y. & Paluri, M. A closer look at spatiotemporal
convolutions for action recognition. *CVPR*, 6450–6459 (2018). https://arxiv.org/abs/1711.11248

Hara, K., Kataoka, H. & Satoh, Y. Can spatiotemporal 3D CNNs retrace the history of 2D CNNs and
ImageNet? *CVPR*, 6546–6555 (2018). https://doi.org/10.1109/CVPR.2018.00685

He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. *CVPR* (2016).
https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

## Kinetics-400 pretraining

Kay, W. et al. The Kinetics human action video dataset. arXiv:1705.06950 (2017).
https://arxiv.org/abs/1705.06950

The implementation uses the official Torchvision video models, e.g.
[`torchvision.models.video.r3d_18`](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.video.r3d_18.html),
with the `KINETICS400_V1` weight enums.

## Single-modality validation in a multimodal Data Descriptor

Bocus, M. J. et al. OPERAnet, a multimodal activity recognition dataset acquired from radio frequency
and vision-based sensors. *Scientific Data* **9**, 474 (2022). https://doi.org/10.1038/s41597-022-01573-2
