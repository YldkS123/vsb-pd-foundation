# -*- coding: utf-8 -*-
"""Insert baseline-rebuild + ablation sections into the research report."""
from pathlib import Path

path = Path("docs/research_report.md")
s = path.read_text(encoding="utf-8")

steps_add = """
### 第 11 步：强基线重建（OOF 审计 + 三组特征）

**目标**：在完全相同的发展集 5 折划分上重建强基线，并核验 OOF 预测、正类概率、阈值来源与指标计算，排除“基线选弱”或“指标计算错误”的质疑。

**操作**：
1. 特征集 A：116 维（均值 + 标准差聚合，58×2）
2. 特征集 B：406 维（mean/std/min/max/median/skew/kurtosis 七统计量，58×7）
3. 特征集 C：464 维窗口展平（8 窗 × 58 维，不聚合）
4. 模型：Logistic Regression（C=10, StandardScaler）、Random Forest（200 树, balanced_subsample）、LightGBM（200 树, balanced）
5. 每折保存 OOF 概率（results/baseline_oof/），报告 PR-AUC/ROC-AUC 及 0.5、max-MCC、召回≥0.5/0.8 四档阈值指标
6. 划分指纹：与 VSB 模型训练共用 StratifiedGroupKFold(5, seed=42)，折叠分配 SHA-256 指纹写入结果

**产物**：`results/baseline_full_comparison.json`、`results/baseline_oof/*.npz`

---

### 第 12 步：模型消融（编码器 × MIL × 相位交互）

**目标**：在锁定窗口方案（K=8 混合策略）与发展集折叠上，逐层验证三个创新模块的独立贡献。

**消融网格**（同一 5 折、epochs=50、patience=20、batch=64）：
- 编码器：双分支融合 / 纯 CNN / 纯统计特征 MLP
- MIL 聚合：gated_attention / attention / mean / max
- 相位交互：循环对称卷积 / 无交互 / 直接拼接 / 最大概率 / 均值聚合
- 测量级聚合：全部配置统一使用 noisy-OR（1 - Π(1-p)）

**产物**：`results/ablations/dev_k8/ablation_summary.json`（16 个配置，含每折 PR-AUC、OOF 概率、参数量）

---

### 第 13 步：鲁棒性测试与最终盲测（待完成）

- 加性噪声、幅值缩放、时间偏移、缺失相位
- 参数量与端到端推理时间
- measurement 级 bootstrap 置信区间
- 在全部消融决策锁定后，仅对 423 盲测集评估一次

"""

s = s.replace("## 二、研究流程（共 10 步）", "## 二、研究流程（共 13 步）")

end_10 = s.find("## 三、实验结果")
if end_10 < 0:
    raise SystemExit("anchor not found")
s = s[:end_10] + steps_add + "\n" + s[end_10:]

baseline_section = """

### 3.5 强基线重建（三组特征 × 三模型，OOF 阈值审计）

| 特征集 | LR PR-AUC | RF PR-AUC | LightGBM PR-AUC |
|---|---|---|---|
| 116 维（均值+标准差） | 0.144 | 0.146 | 0.149 |
| 406 维（七统计量） | 0.128 | 0.171 | 0.188 |
| 464 维（窗口展平） | 0.099 | 0.180 | **0.196** |

- 最强基线：展平特征 + LightGBM，OOF PR-AUC 0.196（ROC-AUC 0.537，F1@maxMCC 0.280）
- 所有 OOF 概率已保存（`results/baseline_oof/`），阈值均从开发集 OOF 选择，无盲测参与
- 划分指纹与模型训练完全一致，消除“基线在别的折上训练”的可能质疑

### 3.6 模型消融（同一 5 折，均值 PR-AUC ± 标准差）

**编码器消融**（gated_attention + 循环相位）：

| 编码器 | 相位级 PR-AUC | 测量级 PR-AUC | 参数量 |
|---|---|---|---|
| 纯 CNN（深度可分离 1D） | **0.614 ± 0.073** | 0.640 | 178,546 |
| 双分支融合（CNN + 特征 MLP） | 0.595 ± 0.073 | 0.634 | 203,634 |
| 纯统计特征 MLP | 0.208 ± 0.070 | 0.227 | 186,866 |

**MIL 聚合消融**（双分支 + 循环相位）：

| 聚合 | 相位级 PR-AUC | 参数量 |
|---|---|---|
| Mean | **0.611 ± 0.092** | 203,376 |
| Attention | 0.603 ± 0.065 | 203,505 |
| Gated-Attention | 0.595 ± 0.073 | 203,634 |
| Max | 0.559 ± 0.045 | 203,376 |

**相位交互消融**（双分支 + gated_attention）：

| 相位交互 | 相位级 PR-AUC | 参数量 |
|---|---|---|
| Max 概率交互 | **0.620 ± 0.039** | 105,330 |
| Mean 聚合交互 | 0.612 ± 0.070 | 105,330 |
| 直接拼接 | 0.599 ± 0.014 | 154,866 |
| 循环对称卷积（参考模型） | 0.595 ± 0.073 | 203,634 |
| 无交互 | 0.474 ± 0.087 | 105,330 |

**结论**：
1. 相位交互是最大贡献项（+0.12~0.15 PR-AUC），验证了三相感知融合的必要性
2. 纯 CNN 编码器与简单 Mean/Max 交互在均值上不低于复杂模块，说明“轻量化优先”成立；但折间标准差较大（±0.04~0.09），差异需在盲测前的最终决策中结合稳定性和 bootstrap CI 综合判断
3. 复杂循环等变与门控注意力不构成明显增益，论文主线应以“覆盖感知窗口采样 + 层级弱监督 + 相位感知交互”为主，自监督/复杂结构作为可选扩展

"""

anchor2 = "## 四、环境与实现"
idx2 = s.find(anchor2)
if idx2 < 0:
    raise SystemExit("env anchor not found")
s = s[:idx2] + baseline_section + "\n" + s[idx2:]

path.write_text(s, encoding="utf-8")
print("report updated, new length:", len(s))
