# AXHome-MM-v1 Video-only 基线

该目录提供 AXHome-MM-v1 的独立视频模态技术验证基线。模型使用 Kinetics-400 预训练的 Torchvision 视频骨干（由 `--arch` 在 `r3d_18`、`mc3_18`、`r2plus1d_18` 中选择，默认 `r3d_18`），对 6,960 个有效人体动作视频进行 14 类分类；64 个 `person_id=NONE` 的 `background_idle` 样本不进入主实验。

这是一条可复现的发布数据验证管线，不是视频动作识别 SOTA，也不包含 CSI–Video 融合。当前 Video 结果接近饱和；增加骨干的作用是检验该饱和是否为单一架构的产物，不是为了比较架构优劣。

## 固定实验规格

- 输入：每段视频在完整同步窗口内均匀抽取 16 帧；
- 缓存：RGB、`uint8`、`(16, 128, 171, 3)`；
- 模型：`torchvision.models.video.{r3d_18, mc3_18, r2plus1d_18}`，由 `--arch` 选择；
- 权重：对应的 `KINETICS400_V1`（`R3D_18_Weights` / `MC3_18_Weights` / `R2Plus1D_18_Weights`）；
- 三个骨干共用同一 `(3, 16, 112, 112)` clip 输入、同一划分与同一训练预算；并列报告用于说明视频窗口的可分性不依赖单一架构选择，不构成模型排行榜；
- 训练裁剪：随机 `112 × 112`，水平翻转概率 0.5；
- 验证/测试裁剪：中心 `112 × 112`；
- 优化器：AdamW，学习率 `1e-4`，权重衰减 `1e-4`；
- 最大 50 epochs，验证 Macro-F1 早停，patience 10；
- 批量大小 16，CUDA 上默认启用 AMP；
- 指标：Accuracy、Balanced Accuracy、Macro-F1、Weighted-F1；
- 汇总：均值、总体标准差（`ddof=0`）、最小值、最大值；
- 混淆矩阵：每次运行先按真实类别逐行归一化，再跨运行求平均。

## 两套协议

### In-domain 五种子

种子固定为 2026–2030。数据按 `(person_id, action_id)` 分层，单层内按 `sample_id` 排序后由当前种子打乱，再按 65%/17.5%/17.5% 分为训练、验证和测试集。每个种子的规模固定为：

```text
train       4,547
validation  1,192
test        1,221
total       6,960
```

正式论文运行必须提供 `--reference-assignment-root`，程序会用 `(sample_id, split, seed)` 逐行核对已经完成的 CSI 审计表。这样 CSI-only 与 Video-only 的 in-domain 结果使用完全相同的样本划分。

### Subject-wise LOSO 十折

十名被试轮流作为测试被试，其下一名被试作为验证被试，其余八名用于训练：`P01→P02`、`P02→P03`、…、`P09→P10`、`P10→P01`。每折固定种子 2026。同一被试的所有环境、会话和试次不会跨集合。

## 环境

AutoDL 5090 已有可用的 CUDA PyTorch 时，先保留其匹配的 `torch`/`torchvision`，只补充缺少的 PyAV、Pillow、NumPy 和 Matplotlib。完整依赖边界记录在 `requirements.txt`。

```bash
cd /root/autodl-tmp/baseline-video-r3d18
/root/miniconda3/bin/python -m pip install -r requirements.txt
```

建议先确认：

```bash
/root/miniconda3/bin/python -c "import torch, torchvision, av; print(torch.__version__, torchvision.__version__, av.__version__, torch.cuda.is_available())"
```

## 运行步骤

### 1. 检查发布清单

```bash
/root/miniconda3/bin/python -B inspect_dataset.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --validate-paths \
  --output /root/autodl-tmp/axhome-video-dataset-profile.json
```

### 2. 一次性构建视频缓存

缓存支持断点复用；已经存在但形状、类型或配置不匹配的文件会被拒绝。

```bash
export OMP_NUM_THREADS=8
/root/miniconda3/bin/python -u -B build_cache.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-video-cache \
  --workers 8
```

成功条件是 `cache_build_report.json` 中 `requested=6960`、`completed=6960`、`failed=[]`、`complete=true`。

### 3. 运行 in-domain 五种子

```bash
/root/miniconda3/bin/python -u -B run_in_domain.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-video-cache \
  --reference-assignment-root /root/autodl-tmp/axhome-results/in_domain \
  --output-dir /root/autodl-tmp/axhome-video-results/in_domain \
  --seeds 2026 2027 2028 2029 2030 \
  --epochs 50 --batch-size 16 --num-workers 8 \
  --device cuda --amp
```

### 4. 运行十折 LOSO

```bash
/root/miniconda3/bin/python -u -B run_loso.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-video-cache \
  --output-dir /root/autodl-tmp/axhome-video-results/loso \
  --epochs 50 --batch-size 16 --num-workers 8 \
  --seed 2026 --device cuda --amp
```

可用 `--test-people P01 P02` 只运行指定折进行调试；这不是正式十折结果。

### 单次调试入口

```bash
# 一个 in-domain 种子
/root/miniconda3/bin/python -u -B train_in_domain.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-video-cache \
  --reference-assignment-root /root/autodl-tmp/axhome-results/in_domain \
  --output-dir /root/autodl-tmp/axhome-video-results/debug/seed_2026 \
  --seed 2026 --epochs 1 --device cuda --amp

# 一个 LOSO 折
/root/miniconda3/bin/python -u -B train_fold.py \
  --dataset-root /root/autodl-tmp/AXHome-MM-v1 \
  --cache-dir /root/autodl-tmp/axhome-video-cache \
  --output-dir /root/autodl-tmp/axhome-video-results/debug/test_P01_val_P02 \
  --test-person P01 --val-person P02 \
  --seed 2026 --epochs 1 --device cuda --amp
```

## 输出与失败恢复

每个种子或折包含 checkpoint、运行配置、划分审计表、训练历史、逐样本预测、分类报告和混淆矩阵。批量目录另外包含运行表、汇总 JSON 和平均归一化混淆矩阵。

`run_config.json` 会留存实际 Python、NumPy、PyTorch、Torchvision、PyAV、Pillow、CUDA/cuDNN、GPU 型号，以及 Kinetics-400 官方 checkpoint 文件名、来源 URL 和 SHA-256；论文复现实验应连同该文件一起保存。

批量入口会捕获单个种子/折的异常，保留其他成功结果，并在每次运行结束后刷新部分汇总。只要有失败，`complete=false` 且进程最终返回非零；不得把部分结果当作正式论文结果。

如果显存不足，优先把 `--batch-size` 从 16 降为 8 或 4。不要改变 16 帧、112 裁剪或数据划分来规避显存问题。

## 论文报告口径

主表应分别列出 CSI-only 与 Video-only 的 in-domain 五种子和 subject-wise LOSO 十折，共四行。两种模态只在相同协议下并列；in-domain 表示同域数据可学习性，LOSO 表示未知被试泛化，二者不能直接视为相同难度。

该实现的研究依据和可引用文献见 `REFERENCES.md`。
