# -*- coding: utf-8 -*-
"""Append window-policy ablation section to the research report."""
from pathlib import Path

path = Path("docs/research_report.md")
s = path.read_text(encoding="utf-8")

if "### 3.9 窗口策略消融" in s:
    print("already present, skip")
    raise SystemExit(0)

section = """

### 3.9 窗口策略消融（K 值与窗口组成，同一参考模型）

在完全相同的开发集 5 折与参考模型（双分支 + gated-attention + 循环相位 + noisy-OR）下比较窗口策略（混合 = 等距 + 事件，逐折均值 PR-AUC ± 标准差）：

| 策略 | K | 组成 | 相位级 PR-AUC | 测量级 PR-AUC |
|---|---|---|---|---|
| mixed_k4 | 4 | 2 等距 + 2 事件 | 0.461 ± 0.070 | 0.523 |
| mixed_k8（锁定主线） | 8 | 4 等距 + 4 事件 | **0.590 ± 0.081** | 0.621 |
| mixed_k12 | 12 | 6 等距 + 6 事件 | （提取与训练进行中） | — |

- K=4 相对 K=8 下降约 0.13 PR-AUC，说明 4 个窗口的覆盖不足
- K=12 结果待 K=12 提取/训练完成后补入
- 单峰 / 纯等距 / 纯事件策略的提取管线已就绪（configs/policy_*.json），作为后续补充实验

"""

anchor = "## 四、环境与实现"
idx = s.find(anchor)
if idx < 0:
    raise SystemExit("env anchor not found")
s = s[:idx] + section + "\n" + s[idx:]

path.write_text(s, encoding="utf-8")
print("report updated, new length:", len(s))
