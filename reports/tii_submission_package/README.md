# IEEE TII 投稿包（Paper-TII）

> 目标：IEEE Transactions on Industrial Informatics（IF≈11，中科院1区Top）
> 状态：**独立投稿**（2026-08-25 决策：不投 TIM，仅投 TII——简化合规，论文已完全自包含）
> 论文完整包含：CAS 采样 + TFE 时频编码 + LHWSD 弱监督 + 可信评估 + 多数据集验证

---

## 文件清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `docs/paper_tii_v2.md` | TII 版论文 v2（完整版，已移除 TIM 互引，完全独立） | ✅ |
| `docs/paper_tii_ieee.tex` | IEEEtran LaTeX 源文件（可编译） | ✅ |
| `docs/figures_tii/` | 4 张图（框架图/采样率/编码器/标注成本） | ✅ |
| `reports/tii_submission_package/cover_letter.md` | 封面信（已移除 TIM 披露段） | ✅ |
| `reports/tii_experiment_tracking.md` | 实验跟踪（4 项全部完成） | ✅ |
| `scripts/make_tii_figures.py` + `make_tii_fig1.py` | 图表生成脚本 | ✅ |

## 核心实验数据（全部真实）

| 实验 | 结果 |
|---|---|
| 编码器对比 | TFE **0.703**±0.025 / simple_cnn 0.615 / LPT 0.588（VSB） |
| 采样率曲线 | 40→5MHz 保留 **83%**（8× 硬件成本降低） |
| 标注比例 | 50% 标注保留 **88%** 性能 |
| VICReg 自监督 | 诚实负结果（−0.017~−0.070） |
| 多数据集 | LPT 外部数据集 0.998（数据规模假设）；28523090 跨设备历史结果 |

## 待办（投稿前）
- [ ] 作者信息/致谢/ORCID
- [ ] IEEEtran 编译验证（需 TeX 环境）
- [ ] 28523090 数据重下载（换网络后补 E4 三臂，可选）
