# -*- coding: utf-8 -*-
"""Fill in mixed_k12 window-policy ablation results in report section 3.9."""
from pathlib import Path

path = Path("docs/research_report.md")
s = path.read_text(encoding="utf-8")

old_row = "| mixed_k12 | 12 | 6 等距 + 6 事件 | （提取与训练进行中） | — |"
new_row = "| mixed_k12 | 12 | 6 等距 + 6 事件 | **0.644 ± 0.092** | 0.668 |"
assert old_row in s, "k12 table row not found"
s = s.replace(old_row, new_row)

old_bullet = "- K=12 结果待 K=12 提取/训练完成后补入"
new_bullet = (
    "- K=12 相对 K=8 提升约 +0.054 PR-AUC（0.644 ± 0.092 vs 0.590 ± 0.081），"
    "更多窗口覆盖带来稳定增益，但窗口数与提取/训练成本约增加 1.5 倍\n"
    "- K=12 是否取代 K=8 作为锁定主线，待与最终模型/盲测方案一并确认"
)
assert old_bullet in s, "k12 bullet not found"
s = s.replace(old_bullet, new_bullet)

path.write_text(s, encoding="utf-8")
print("report 3.9 updated, new length:", len(s))
