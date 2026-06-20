# PPM-AD：可塑性原型记忆工业异常检测实验

本仓库用于验证课程中期方案《从 PatchCore 到可塑性原型记忆：面向少样本工业缺陷检测的类脑改进方案》。实现保留 PatchCore 的冻结预训练特征和 patch 级定位框架，并加入：

1. 正常原型及其局部半径；
2. 原型竞争熵与胜者优势；
3. 吸引子式状态迭代；
4. 局部群体协同；
5. 高置信门控的可塑性更新。

代码目标是产出可直接用于最终报告的原始结果，而不是追求一个不可审计的单一最优数值。每次运行都会保存逐类别、逐 shot、逐随机种子的结果。

## 1. 实验与研究问题的对应关系

| 实验 | 命令模式 | 主要回答的问题 |
|---|---|---|
| 少样本主实验 | `fewshot` | 5/10/20-shot 下是否优于 PatchCore、压缩 memory 和 prototype-only |
| 模块消融 | `ablation` | 竞争、吸引子、局部协同是否各自提供有效贡献 |
| 分布漂移与污染 | `drift` | 固定、无门控、门控更新在渐进漂移下的适应性和污染风险 |
| 原型数与效率 | `k_sweep` | K 对性能、显存/内存和推理时间的影响 |

核心指标包括 Image-AUROC、Image-AP、Pixel-AUROC、Pixel-AP、AUPRO@0.3、max-F1、IoU、热力图碎片化、单图推理时间和记忆占用。漂移实验额外记录异常分数下降量 `delta_anomaly_score` 和实际写入数 `accepted_updates`。

## 2. 服务器环境

建议使用 Linux、Python 3.10 或 3.11、CUDA 11.8 以上及至少 12 GB 显存。代码也支持 CPU，但完整实验会很慢。

```bash
git clone https://github.com/gubeihai123/sad.git
cd sad

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果服务器需要安装特定 CUDA 版本的 PyTorch，请先按服务器 CUDA 环境安装对应的 `torch` 和 `torchvision`，再执行：

```bash
pip install -r requirements.txt
```

首次运行会由 `torchvision` 下载 ImageNet 预训练的 WideResNet-50-2 权重。实验不使用缺陷图像训练模型。

## 3. 准备 MVTec AD

下载并解压 MVTec AD，使目录结构为：

```text
data/mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/*.png
│   ├── test/good/*.png
│   ├── test/<defect>/*.png
│   └── ground_truth/<defect>/*_mask.png
├── carpet/
└── ...
```

也可以把数据放在任意位置，然后修改 `configs/default.yaml` 的 `data_root`。数据集本身不要提交到 GitHub。

## 4. 先做本地/服务器自检

```bash
pytest -q
```

再复制一份最小配置，只保留 1 个类别、1 个 shot、1 个 seed，并把 K 调小：

```bash
cp configs/default.yaml configs/smoke.yaml
# 编辑 smoke.yaml：categories: [bottle]，shots: [5]，seeds: [0]，prototypes: 64
python run_experiments.py --config configs/smoke.yaml --mode fewshot --run-name smoke
```

确认 `outputs/smoke/results_raw.csv` 非空后再开始完整实验。

## 5. 推荐运行顺序

不要一开始直接运行 `all`。按以下顺序运行，任一阶段失败时不会丢失之前结果：

```bash
python run_experiments.py --mode fewshot  --run-name fewshot-main
python run_experiments.py --mode ablation --run-name ablation-main
python run_experiments.py --mode drift    --run-name drift-main
python run_experiments.py --mode k_sweep  --run-name k-sweep-main
```

确认资源足够时也可执行：

```bash
python run_experiments.py --mode all --run-name final-all
```

默认配置包含 10 个 MVTec 类别、3 个 shot 和 3 个随机种子。若时间不足，优先保证：

1. `fewshot` 的 5/10/20-shot、3 seeds；
2. `ablation` 的 10-shot、3 seeds；
3. `drift` 至少覆盖 2 个纹理类和 2 个物体类；
4. `k_sweep` 可先只用 `carpet`、`tile`、`bottle`、`capsule`。

## 6. 输出文件

每个运行目录包含：

```text
outputs/<run-name>/
├── config.yaml             # 实际运行配置快照
├── results_raw.csv         # 每个类别/shot/seed 的原始结果，后续统计以此为准
├── results_summary.csv     # 自动聚合的均值和标准差
├── run_manifest.json       # PyTorch、CUDA、设备和完成行数
└── heatmaps/               # 少量定性可视化
```

程序每完成一个模型就增量写入 CSV，因此长任务被中断时已完成的行仍保留。重新运行时请使用新的 `--run-name`，避免覆盖旧结果。

## 7. 跑完后上传结果

先检查行数和异常值：

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
for p in Path('outputs').glob('*/results_raw.csv'):
    df = pd.read_csv(p)
    print(p, 'rows=', len(df), 'NaN=', int(df.isna().sum().sum()))
PY
```

然后提交代码生成的结果目录。不要提交 MVTec 数据或预训练权重。

```bash
git add outputs/
git commit -m "Add PPM-AD experiment results"
git push
```

建议同时记录服务器 GPU 信息，便于解释效率数据：

```bash
nvidia-smi > outputs/gpu-info.txt
```

结果推送后，需要进一步分析的最小文件集合是各运行目录中的 `config.yaml`、`results_raw.csv`、`run_manifest.json` 和代表性热力图。

## 8. 实现说明与限制

- `patchcore` 使用完整（受 `max_memory_patches` 上限约束）正常 patch memory；`patchcore-coreset` 使用固定比例、固定种子的随机压缩 memory，作为透明的压缩基线。
- `prototype-only` 仅使用 K-means 原型距离，不启用竞争项、吸引子迭代和局部协同。
- AUPRO 使用 FPR 不超过 0.3 的数值积分近似；最终报告中应明确此实现口径。
- 漂移由亮度、对比度和高斯模糊的渐进组合构造。无门控策略会吸收测试流中的所有 patch，用于暴露最坏情况下的污染；门控策略限制每批更新比例。
- 当前代码以清楚、可消融、可复现为优先。若需要与 PatchCore 官方实现的论文数字严格对齐，应额外运行官方仓库作为外部参考，不应把本仓库的轻量基线声称为官方完全复现。


