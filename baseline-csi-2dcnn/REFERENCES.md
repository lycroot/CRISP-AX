# 基线参考文献与实现边界

检索和核对日期：2026-07-17。以下只记录对当前程序有直接影响的论文、项目文档和官方实现说明。

## 1. EHUNAM Scientific Data 数据描述论文

de Armas, E., Diaz, G., Sobron, I., et al. *EHUNAM, a WiFi CSI-based dataset for human and machine sensing*. Scientific Data, 12, 1950 (2025). DOI: [10.1038/s41597-025-06238-4](https://doi.org/10.1038/s41597-025-06238-4)。

论文原文：[Nature / Scientific Data](https://www.nature.com/articles/s41597-025-06238-4)。

论文的技术验证模型包括：

- 3 个 2D 卷积层：32 个 `5 x 5`、64 个 `3 x 3`、128 个 `3 x 3` 卷积核；
- 256 和 128 单元的全连接层；
- Mish、MaxPooling、Dropout、Flatten 和 Softmax；
- Adam，学习率为 0.001；
- 输入为 25 组 CSI，幅度和相位作为 2 个通道；
- 同一接收机—应用组合内按 65%/17.5%/17.5% 划分训练、验证和测试。

本程序借鉴其 3 层卷积通道数、卷积核、Mish 和 Adam。没有照搬以下部分：

- AXHome-MM-v1 使用双 RX 幅度作为输入通道，不使用未经校准的原始相位；
- 使用全局平均池化控制 996 子载波输入对应的参数量，不使用大规模 Flatten；
- 使用 subject-wise LOSO，不使用可能混合同一被试和会话的随机样本拆分；
- 输入覆盖完整动作窗口，经等距选取得到 256 个 packet，而不是从 60 秒测量中构造 25-CSI 小组。

因此，本文可将该模型描述为「EHUNAM-inspired lightweight 2D CNN baseline」，不能写成对 EHUNAM 模型的完全复现。

## 2. FeitCSI 官方格式与解析器

Hutar, M., Brida, P., and Machaj, J. *FeitCSI, the 802.11 CSI tool* (2023). 项目主页：[FeitCSI](https://feitcsi.kuskosoft.com/)。

- 官方格式：[CSI format](https://feitcsi.kuskosoft.com/csi_format/)
- 官方 Python 示例：[Python parser](https://feitcsi.kuskosoft.com/python/)

官方文档规定：

- 每条记录包含固定 272-byte header 和可变长 CSI payload；
- header 的 0–3 byte 保存 payload 大小；
- 46、47 byte 分别保存 RX 和 TX 数；
- 52–55 byte 保存子载波数；
- 每个复数 CSI 值用 4 bytes 表示，即 signed `int16` 实部和 signed `int16` 虚部；
- payload 大小应为 `4 × RX × TX × subcarriers`。

`axhome_csi/feitcsi.py` 按上述布局实现流式、little-endian 解析，并额外检查截断记录、尺寸矛盾和 packet 间形状变化。代码没有复制第三方大段实现，只按公开格式重写，并用正式 ZIP 样本核对。

## 3. 跨人物与跨环境评估依据

Meneghello, F., Garlisi, D., Dal Fabbro, N., Tinnirello, I., and Rossi, M. *SHARP: Environment and Person Independent Activity Recognition With Commodity IEEE 802.11 Access Points*. IEEE Transactions on Mobile Computing, 22, 6160–6175 (2023). DOI: [10.1109/TMC.2022.3185681](https://doi.org/10.1109/TMC.2022.3185681)。预印本：[arXiv:2103.09924](https://arxiv.org/abs/2103.09924)。

该工作明确把跨未知人物和环境的泛化作为 WiFi HAR 的核心问题。它不是本程序 CNN 架构的直接来源，但支持将 AXHome-MM-v1 的主验证协议设为被试独立划分，而不是只报告同域随机拆分结果。

## 4. Mish 激活函数

Misra, D. *Mish: A Self Regularized Non-Monotonic Activation Function*. BMVC 2020. DOI: [10.5244/C.34.191](https://doi.org/10.5244/C.34.191)。预印本：[arXiv:1908.08681](https://arxiv.org/abs/1908.08681)。

本程序使用 PyTorch 内置 `torch.nn.Mish`，与 EHUNAM 的中间层激活保持一致。

## 5. 类别加权交叉熵

PyTorch 官方文档：[CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。官方文档说明 `weight` 可对各类别损失进行重标定，适用于不平衡训练集。

本程序使用经典 balanced 权重：

```text
weight_c = N / (C × count_c)
```

其中 `N` 是当前训练折样本数，`C` 是类别数，`count_c` 是当前训练折中类别 `c` 的样本数。验证集和测试集不参与权重计算。

