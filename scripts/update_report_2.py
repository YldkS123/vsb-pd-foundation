# -*- coding: utf-8 -*-
"""Append metric-convention audit + robustness sections to the research report."""
from pathlib import Path

path = Path("docs/research_report.md")
s = path.read_text(encoding="utf-8")

if "### 3.7 指标口径审计" in s:
    print("sections already present, skip")
    raise SystemExit(0)

audit_section = """

### 3.7 指标口径审计（OOF 合并 vs 逐折均值）

由于逐折均值 PR-AUC 与 OOF 合并 PR-AUC 可能不同，特对全部模型在同一 5 折上按两种口径复核：

| 模型 | OOF 合并 PR-AUC | 逐折均值 PR-AUC | OOF 合并 ROC | 逐折均值 ROC |
|---|---|---|---|---|
| agg116 LR | 0.144 | 0.156 | 0.527 | 0.528 |
| agg116 RF | 0.146 | 0.153 | 0.556 | 0.556 |
| agg116 LGBM | 0.149 | 0.157 | 0.531 | 0.530 |
| agg406 RF | 0.171 | 0.180 | 0.558 | 0.558 |
| agg406 LGBM | 0.188 | 0.198 | 0.536 | 0.537 |
| flatten RF | 0.180 | 0.184 | 0.557 | 0.557 |
| flatten LGBM（最强基线） | 0.196 | 0.205 | 0.537 | 0.537 |
| **VSB MIL（参考模型）** | **0.495** | **0.590** | **0.908** | **0.935** |

**结论**：无论采用哪种口径，VSB MIL 均大幅领先最强基线：
- OOF 合并口径：+0.30（+153%）
- 逐折均值口径：+0.385（+188%）
ROC-AUC 差距约 +0.37。报告的 0.586/0.590 为逐折均值口径；基线对比使用同一口径时结论不变。

### 3.8 鲁棒性测试（参考模型，仅推理端扰动）

对 5 折参考模型施加推理端扰动，评估相位级与测量级 PR-AUC 相对下降：

| 扰动 | 相位级 PR-AUC | 相对下降 | 测量级 PR-AUC | 相对下降 |
|---|---|---|---|---|
| 基线（无扰动） | 0.426* | — | 0.472 | — |
| 高斯噪声 20 dB | 0.370 | -13% | 0.474 | 0% |
| 高斯噪声 10 dB | 0.207 | -51% | 0.274 | -42% |
| 高斯噪声 5 dB | 0.138 | -68% | 0.191 | -60% |
| 幅值 ×0.8 | 0.426 | 0% | 0.472 | 0% |
| 幅值 ×1.2 | 0.426 | 0% | 0.472 | 0% |
| 时间偏移 -64 | 0.246 | -42% | 0.296 | -37% |
| 时间偏移 +64 | 0.227 | -47% | 0.287 | -39% |
| 时间偏移 -128 | 0.367 | -14% | 0.435 | -8% |
| 时间偏移 +128 | 0.360 | -15% | 0.421 | -11% |
| 缺失相位 A | — | — | 0.472 | -0.1% |
| 缺失相位 B | — | — | 0.471 | -0.4% |
| 缺失相位 C | — | — | 0.472 | -0.2% |

*基线为 results/model_full 检查点 OOF 合并口径（训练随机性导致与 3.7 表略异，相对下降不受影响）。

**结论**：
1. 幅值缩放与缺失单相几乎不影响性能（noisy-OR 三相冗余有效）
2. 20 dB 噪声仍保持 -13%，5 dB 严重噪声下降 68%（符合预期，可作为论文的局限性）
3. 测量级 bootstrap 95% CI：[0.415, 0.529]（中位 0.475）
4. 参考模型参数量 203,634；GPU 端到端推理 0.7 ms/测量（batch=64，46.4 ms/批）

"""

anchor2 = "## 四、环境与实现"
idx2 = s.find(anchor2)
if idx2 < 0:
    raise SystemExit("env anchor not found")
s = s[:idx2] + audit_section + "\n" + s[idx2:]

path.write_text(s, encoding="utf-8")
print("report updated, new length:", len(s))
