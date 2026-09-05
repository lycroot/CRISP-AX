# AXHome-MM-v1 CSI-only Baselines

这是 AXHome-MM-v1 的 CSI-only 技术验证基线。程序直接读取正式发布包中的 `manifests/archive_index.csv` 和 FeitCSI `.dat` 文件，完成 14 类人体动作识别，并提供同域可学习性验证与未知被试泛化评估两套协议。

## 基线结论

- **输入：** 双 RX 的 CSI 幅度时—子载波图，形状为 `(2, 256, 128)`。
- **模型：** 由 `--arch` 选择的四个架构之一，全部消费同一冻结输入、同一划分与同一训练预算（见“多架构技术验证”）。默认 `cnn2d` 为 3 层轻量 2D CNN，卷积通道 32、64、128，分类头 256 和 128 单元。
- **任务：** 14 类有人体动作识别；主实验不包含 64 个无人物身份的 `background_idle` 样本。
- **协议 A（同域技术验证）：** 按 `(person_id, action_id)` 分层，以 65%/17.5%/17.5% 划分训练、验证和测试集，固定运行 2026–2030 五个随机种子。
- **协议 B（未知被试泛化）：** 10 折 subject-wise LOSO。每折 1 名测试被试、1 名验证被试、其余 8 名训练被试。
- **不平衡处理：** 类别权重只根据当前训练折计算，不读取验证集或测试集统计量。
- **主要指标：** Accuracy、Balanced Accuracy、Macro-F1、Weighted-F1、各类 F1 和混淆矩阵。

两套协议使用相同的预处理、模型、类别权重、早停条件和指标。设计借鉴了 EHUNAM 在 Scientific Data 数据描述论文中使用的三层 2D CNN，但没有照搬其相位输入；协议 A 用于展示常规同域条件下的数据可学习性，协议 B 用于评估更严格的未知被试泛化。具体差异见 [REFERENCES.md](./REFERENCES.md)。

## 目录

```text
baseline-csi-2dcnn/
├── axhome_csi/          # 解析、预处理、划分、模型、训练和评估
├── tests/               # 二进制解析、划分、缓存、指标和训练冒烟测试
├── build_cache.py       # 可选：预先生成幅度图缓存
├── inspect_dataset.py   # 检查发布清单和数据分布
├── train_in_domain.py   # 运行一个同域随机种子
├── run_in_domain.py     # 顺序运行并汇总五个同域随机种子
├── train_fold.py        # 训练一个被试独立折
├── run_loso.py          # 运行全部 10 折
├── DATASET_PROFILE.md   # 本次发布数据的实际分布记录
├── REFERENCES.md        # 文献与实现边界
└── requirements.txt
```

## AutoDL 快速开始

### 1. 上传并解压数据

将本目录和 `AXHome-MM-v1.zip` 上传到 AutoDL。建议把大数据放到 `/root/autodl-tmp`：

```bash
mkdir -p /root/autodl-tmp/axhome-data
unzip AXHome-MM-v1.zip -d /root/autodl-tmp/axhome-data
```

解压后，数据根目录应满足：

```text
/root/autodl-tmp/axhome-data/AXHome-MM-v1/
├── data/
├── manifests/archive_index.csv
├── reports/
└── DATASET_VERSION.txt
```

程序目前面向解压后的发布目录，不直接从 46 GB ZIP 中随机读取训练样本。

### 2. 安装环境

建议创建 AutoDL 的 PyTorch 镜像实例，然后执行：

```bash
cd /root/baseline-csi-2dcnn
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

推荐 Python 3.10–3.12、PyTorch 2.2 以上版本。若 AutoDL 镜像已经包含匹配 CUDA 的 PyTorch，可只补装 NumPy 和 Matplotlib，避免替换镜像自带的 CUDA 版本：

```bash
python -m pip install "numpy>=1.26,<3" "matplotlib>=3.8,<4"
```

程序已在 5090 环境现有的 NumPy 2.x 上正常完成缓存和 LOSO 运行，因此不需要为了本基线主动降级 NumPy；`requirements.txt` 同时兼容 NumPy 1.26 和 2.x。

### 3. 检查数据

```bash
python inspect_dataset.py \
  --dataset-root /root/autodl-tmp/axhome-data/AXHome-MM-v1 \
  --validate-paths \
  --output /root/autodl-tmp/axhome-profile.json
```

正式发布包应报告 7,024 个样本，其中 6,960 个人体动作样本、64 个背景样本。
（历史记录：排除前的口径为 7,025 / 65，其中 `S01_NONE_background_idle_cross_link_E3_faucet_on_water_flow_sink_R001` 因 CSI–视频时间错位被排除；该样本属于无人背景，人体动作基线的 6,960 条输入不受影响。）

### 4. 建立预处理缓存

缓存不是必需项，但完整 10 折会反复访问同一数据，建议先建立缓存：

```bash
python build_cache.py \
  --dataset-root /root/autodl-tmp/axhome-data/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-cache \
  --workers 8
```

默认缓存约包含 6,960 个 `(2, 256, 128)` 的 `float32` 数组，未压缩理论数据量约为 1.70 GiB。脚本会校验解析 packet 数是否与 manifest 一致；任何失败都会写入 `cache_build_report.json` 并返回非零退出码。

可以先检查 20 个样本：

```bash
python build_cache.py \
  --dataset-root /root/autodl-tmp/axhome-data/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-cache \
  --workers 2 \
  --limit 20
```

### 5. 运行协议 A：五种子同域技术验证

正式实验使用按“被试 × 动作”分层的 65%/17.5%/17.5% 划分。每个种子独立完成样本划分、模型初始化、训练、早停和测试：

```bash
export OMP_NUM_THREADS=8
/root/miniconda3/bin/python -u -B run_in_domain.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-cache \
  --output-dir /root/autodl-tmp/axhome-results/in_domain \
  --seeds 2026 2027 2028 2029 2030 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

只检查一个种子时可运行：

```bash
/root/miniconda3/bin/python -u -B train_in_domain.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-cache \
  --output-dir /root/autodl-tmp/axhome-results/in_domain_smoke_seed_2026 \
  --seed 2026 \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

真实发布清单下，每个种子的集合大小固定为训练 4,547、验证 1,192、测试 1,221，总计 6,960；不同种子改变逐样本归属，但不改变集合大小。

### 6. 运行协议 B：单折检查

先运行 P01 测试、P02 验证的单折：

```bash
python train_fold.py \
  --dataset-root /root/autodl-tmp/axhome-data/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-cache \
  --output-dir /root/autodl-tmp/axhome-results/test_P01_val_P02 \
  --test-person P01 \
  --val-person P02 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

调试程序和显存时，可以先用 `--epochs 1 --batch-size 8 --num-workers 2`。如果显存不足，优先减小 `--batch-size`，不要改变测试协议。

### 7. 运行协议 B：全部 10 折

```bash
python run_loso.py \
  --dataset-root /root/autodl-tmp/axhome-data/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-cache \
  --output-dir /root/autodl-tmp/axhome-results/loso \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

若只想运行部分折：

```bash
python run_loso.py \
  --dataset-root /root/autodl-tmp/axhome-data/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-cache \
  --output-dir /root/autodl-tmp/axhome-results/partial \
  --test-people P01 P02
```

## LOSO 对应关系

| 折 | 测试被试 | 验证被试 | 训练被试数 |
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

同一被试的 3 个环境、全部会话和试次始终位于同一个集合中，不会跨 train、validation 和 test。

## 预处理细节

1. 按 FeitCSI 官方 272-byte header 解析记录。
2. 将 signed `int16` 实部和虚部还原为复数 CSI，得到 `(packet, RX, TX, subcarrier)`。
3. 验证每个文件的 packet 数与 `csi_packets_written` 一致。
4. 计算 `log1p(abs(CSI))`，不使用未经校准的相位。
5. 根据整个样本的子载波中位能量识别零值或极低能量载波，并沿固定频率轴使用相邻有效载波线性插值修复。
6. 从固定的 996 子载波频率轴上等距选取 128 个位置，并从动作窗口中等距选取 256 个 packet。不同样本的第 `j` 个频率位置保持对应，避免样本自适应删载波造成频率错位。
7. 每个 RX 通道在样本内部做 z-score。

发布样本已经是同步动作窗口，因此程序不使用 FeitCSI header 中本批数据为 0 的 timestamp 字段重新切片。

## 训练输出

每个折目录包含：

- `best_model.pt`：验证集 Macro-F1 最优的 checkpoint；
- `run_config.json`：超参数、类别、设备和人员划分；
- `history.csv`：逐 epoch 训练/验证损失与指标；
- `metrics.json`：测试集总体和逐类指标；
- `classification_report.csv`：逐类 precision、recall、F1 和 support；
- `predictions.csv`：逐样本真实标签和预测标签；
- `confusion_matrix_counts.csv`：混淆矩阵原始计数；
- `confusion_matrix_counts.png`：原始计数图；
- `confusion_matrix_normalized.png`：按真实类别归一化的混淆矩阵图。

完整 LOSO 还会生成 `loso_folds.csv` 和 `loso_summary.json`。论文中建议同时报告 10 折均值与标准差，并公开逐折结果。

每个 in-domain 种子目录还会保存 `split_assignments.csv`，逐条记录 `sample_id`、被试、动作、归档会话、集合归属和种子。五种子根目录生成：

- `in_domain_runs.csv`：逐种子的四个主指标、测试损失和最佳 epoch；
- `in_domain_summary.json`：请求、完成和失败种子，以及 Accuracy、Balanced Accuracy、Macro-F1、Weighted-F1 的均值、总体标准差、最小值和最大值；
- `confusion_matrix_mean_normalized.csv`：五次运行逐行归一化后再平均的混淆矩阵；
- `confusion_matrix_mean_normalized.png`：平均归一化混淆矩阵图。

只有全部请求种子成功时，汇总文件中的 `complete` 才为 `true`。单个种子失败时，程序保留其他已完成结果、写明错误并最终返回非零退出码。

论文中应将协议 A 报告为同域技术可用性验证，将协议 B 报告为未知被试泛化结果。两者评估难度和重采样单位不同，不应直接比较高低或做跨协议显著性检验。五种子结果报告 `mean ± std_population`，LOSO 保留逐折结果及十折统计。

## 测试

无需 PyTorch 的数据层测试：

```bash
python -m unittest discover -s tests -v
```

安装 PyTorch 后，同一命令还会运行模型前向和 1-epoch 合成数据冒烟测试。

## 重要边界

- 该程序是数据可用性技术验证基线，不应描述成最先进模型。
- 多个架构并列报告的目的是说明「可分性不依赖单一架构选择」，不是模型排行榜；各架构没有做逐架构超参搜索，架构之间的名次差异不得解释为方法优劣结论。
- `background_idle` 没有被试身份，因此没有混入 14 类 LOSO 主结果。
- 当前版本不使用相位；在建立并验证相位校准流程后，才能将相位作为额外通道加入公平比较。
- 同域分层结果可以与 LOSO 并列报告，但只能分别解释为“数据可学习性”和“未知被试泛化”，不能描述为同难度模型优劣比较。
