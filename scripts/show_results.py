"""Print experiment results."""
import json
import torch
from pathlib import Path

print("=== Baseline Results ===")
bp = Path("results/baselines_subset200/baseline_results.json")
if bp.exists():
    print(json.dumps(json.loads(bp.read_text()), indent=2, default=str))

print("\n=== Model Results ===")
metrics_list = []
for f in sorted(Path("results/model_subset200").glob("model_fold*.pt")):
    ckpt = torch.load(str(f), map_location="cpu")
    m = ckpt.get("metrics", {})
    print(f"  {f.stem}: acc={m.get('accuracy'):.4f} prec={m.get('precision'):.4f} "
          f"recall={m.get('recall'):.4f} f1={m.get('f1'):.4f} "
          f"pr_auc={m.get('pr_auc','N/A')} roc_auc={m.get('roc_auc','N/A')}")
    metrics_list.append(m)

aucs = [m.get("pr_auc") for m in metrics_list if m.get("pr_auc") is not None]
import numpy as np
if aucs:
    print(f"\n  Mean PR-AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
