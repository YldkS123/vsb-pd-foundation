# -*- coding: utf-8 -*-
"""Record the final model-lock decision (keep K=8) in report sections 3.9 and 5."""
from pathlib import Path

path = Path("docs/research_report.md")
s = path.read_text(encoding="utf-8")

# --- 3.9: replace pending-confirmation bullet with the final decision ---
old_bullet = "- K=12 是否取代 K=8 作为锁定主线，待与最终模型/盲测方案一并确认"
new_bullet = (
    "- 最终决策：保留 K=8（4u+4e）作为锁定主线。K=12 在开发集上最优（0.644），"
    "但为维持「盲测集仅评估一次」的协议纪律（盲测收据已锁定于 K=8 模型），"
    "不再更换最终模型；K=12 结论作为开发集消融证据写入本文\n"
    "- 若后续需要显著性论证，可补充多种子（3 seeds × 5 折）稳定性实验"
    "（脚本 scripts/run_multi_seed.py 已备好）"
)
assert old_bullet in s, "3.9 pending bullet not found"
s = s.replace(old_bullet, new_bullet)

# --- 五、结论: append window-policy ablation conclusion ---
old_sentence = "测量级 PR-AUC 达到 0.609，表明三相关联网罗 OR 策略有效。"
new_sentence = (
    "测量级 PR-AUC 达到 0.609，表明三相融合 + noisy-OR 策略有效。"
    "窗口策略消融进一步支持覆盖感知采样假设：开发集 PR-AUC 随窗口数单调上升"
    "（K=1: 0.255 → K=4: 0.461 → K=8: 0.590 → K=12: 0.644），"
    "且事件窗口携带主要判别信息（纯事件 0.591 ≈ 混合 K=8 0.590，明显高于纯等距 0.526）；"
    "为保持盲测集一次性评估的协议纪律，最终锁定 K=8 作为报告模型，K=12 作为开发集证据。"
)
assert old_sentence in s, "conclusion sentence not found"
s = s.replace(old_sentence, new_sentence)

path.write_text(s, encoding="utf-8")
print("report updated, new length:", len(s))
