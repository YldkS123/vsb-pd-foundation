# -*- coding: utf-8 -*-
"""Complete window-policy ablation table (all six policies) in report 3.9."""
from pathlib import Path

path = Path("docs/research_report.md")
s = path.read_text(encoding="utf-8")

old_block = """| 策略 | K | 组成 | 相位级 PR-AUC | 测量级 PR-AUC |
|---|---|---|---|---|
| mixed_k4 | 4 | 2 等距 + 2 事件 | 0.461 ± 0.070 | 0.523 |
| mixed_k8（锁定主线） | 8 | 4 等距 + 4 事件 | **0.590 ± 0.081** | 0.621 |
| mixed_k12 | 12 | 6 等距 + 6 事件 | **0.644 ± 0.092** | 0.668 |

- K=4 相对 K=8 下降约 0.13 PR-AUC，说明 4 个窗口的覆盖不足
- K=12 相对 K=8 提升约 +0.054 PR-AUC（0.644 ± 0.092 vs 0.590 ± 0.081），更多窗口覆盖带来稳定增益，但窗口数与提取/训练成本约增加 1.5 倍
- K=12 是否取代 K=8 作为锁定主线，待与最终模型/盲测方案一并确认
- 单峰 / 纯等距 / 纯事件策略的提取管线已就绪（configs/policy_*.json），作为后续补充实验"""

new_block = """| 策略 | K | 组成 | 相位级 PR-AUC | 测量级 PR-AUC |
|---|---|---|---|---|
| single | 1 | 1 等距 | 0.255 ± 0.029 | 0.273 |
| equidistant | 8 | 8 等距 + 0 事件 | 0.526 ± 0.105 | 0.534 |
| event | 8 | 0 等距 + 8 事件 | 0.591 ± 0.056 | 0.624 |
| mixed_k4 | 4 | 2 等距 + 2 事件 | 0.461 ± 0.070 | 0.523 |
| mixed_k8（锁定主线） | 8 | 4 等距 + 4 事件 | 0.590 ± 0.081 | 0.621 |
| mixed_k12 | 12 | 6 等距 + 6 事件 | **0.644 ± 0.092** | 0.668 |

- 单峰覆盖严重不足（0.255）；事件窗口携带主要判别信息：纯事件（0.591）与混合 K=8（0.590）相当，均明显高于纯等距（0.526），说明事件采样是信息增益的主要来源
- K=4（2u+2e）相对 K=8 下降约 0.13 PR-AUC，4 个窗口的覆盖不足
- K=12（6u+6e）最优：0.644 ± 0.092，相对 K=8 提升 +0.054，但窗口数与提取/训练成本约增加 1.5 倍
- K=12 是否取代 K=8 作为锁定主线，待与最终模型/盲测方案一并确认"""

assert old_block in s, "3.9 table block not found"
s = s.replace(old_block, new_block)

path.write_text(s, encoding="utf-8")
print("report 3.9 full table written, new length:", len(s))
