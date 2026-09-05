# Video-only R3D-18 基线参考资料

以下资料在 2026-07-18 经官方文档、论文原文或出版方页面核对。本文件区分“模型来源”“预训练数据来源”和“Scientific Data 技术验证先例”，避免把其他数据集的数值误写为 AXHome-MM 的可比结果。

## 1. 3D ResNet / R3D-18

Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. “Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?” *CVPR 2018*, pp. 6546–6555. DOI: [10.1109/CVPR.2018.00685](https://doi.org/10.1109/CVPR.2018.00685). [CVF 论文原文](https://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html).

与本基线的关系：论文系统研究了使用时空三维卷积的 3D ResNet，并验证了在 Kinetics 上训练深层 3D CNN 及把 Kinetics 预训练表示迁移到较小视频数据集的合理性。AXHome-MM 选择 18 层版本作为计算成本与可复现性之间的技术验证基线，不声称它是当前最优视频模型。

残差网络基础来源：Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. “Deep Residual Learning for Image Recognition.” *CVPR 2016*. [CVF 论文原文](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

## 2. Kinetics-400 预训练

Will Kay et al. “The Kinetics Human Action Video Dataset.” arXiv:1705.06950, 2017. [论文页面](https://arxiv.org/abs/1705.06950).

与本基线的关系：Kinetics 提供 400 类大规模人体动作短视频。对 AXHome-MM 的 6,960 个视频直接从零训练 3D CNN 更容易过拟合，因此采用 Kinetics-400 预训练、再微调全部 R3D-18 参数。

实现使用 Torchvision 0.23 的官方接口：[`torchvision.models.video.r3d_18`](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.video.r3d_18.html)，权重枚举固定为 `R3D_18_Weights.KINETICS400_V1`。程序不会在预训练权重不可用时静默退化为随机初始化。

## 3. Scientific Data 多模态技术验证先例

Mohammud J. Bocus et al. “OPERAnet, a multimodal activity recognition dataset acquired from radio frequency and vision-based sensors.” *Scientific Data* 9, 474 (2022). DOI: [10.1038/s41597-022-01573-2](https://doi.org/10.1038/s41597-022-01573-2). [出版方全文](https://www.nature.com/articles/s41597-022-01573-2).

该论文在同一随机划分下分别训练 WiFi CSI、PWR 和 Kinect 模态，并另做简单融合。论文报告的独立模态 Accuracy 为 WiFi CSI 93.5%、PWR 86.5%、Kinect 85.8%，融合为 96.7%。这些数值来自 OPERAnet 自己的 6 类任务、80/20 划分和模型设置，**不能与 AXHome-MM 的 14 类、五种子分层或 LOSO 结果直接比较**。

与本基线的关系：它支持 Scientific Data 数据论文用独立模态基线证明各模态可学习性，并在相同随机划分下公平比较的写法。AXHome-MM 当前先报告 CSI-only 和 Video-only，暂不加入融合；这属于范围控制，不影响多模态数据发布的完整性。

## 4. 论文中建议引用方式

- 方法段：引用 Hara et al. 说明 3D ResNet/R3D-18；引用 Kay et al. 说明 Kinetics-400 预训练数据。
- 实现细节或代码可用性：引用 Torchvision 官方 R3D-18 文档并记录实际 `torch`、`torchvision` 版本和权重枚举。
- 技术验证设计：可引用 OPERAnet，说明多模态数据描述论文可以先分别验证各模态；同时明确其准确率不是 AXHome-MM 的比较基线。
- 结果表：只填写本程序在 AXHome-MM 正式五种子和十折运行得到的均值与总体标准差，不填 Torchvision Kinetics top-1 或 OPERAnet Accuracy。
