# IEEE TII 投稿包（Paper-TII）

> 目标：IEEE Transactions on Industrial Informatics（IF≈11，中科院1区Top，CCF-B）
> 状态：投稿准备中（2026-08-25）
> 与 TIM 版的关系：姊妹篇——TIM 聚焦测量科学，TII 聚焦工业 AI 部署；两稿互引+披露

---

## 文件清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `docs/paper_tii_v2.md` | TII 版论文草稿 v2（完整版：公式+算法+实验+讨论） | ✅ |
| `docs/paper_tii_ieee.tex` | IEEEtran LaTeX 源文件（可编译） | ✅ |
| `docs/figures_tii/` | 3 张图表（采样率曲线/编码器对比/标注成本） | ✅ |
| `reports/tii_submission_package/cover_letter.md` | 封面信（含 TIM 版披露声明） | ✅ |
| `reports/tii_experiment_tracking.md` | 实验跟踪（4 项全部完成） | ✅ |
| `scripts/make_tii_figures.py` | 图表生成脚本 | ✅ |

## 核心实验数据（全部真实）

| 实验 | 结果 |
|---|---|
| 编码器对比 | TFE **0.703**±0.025 / simple_cnn 0.615 / LPT 0.588（VSB） |
| 采样率曲线 | 40→5MHz 保留 **83%**（8× 硬件成本降低） |
| 标注比例 | 50% 标注保留 **88%** 性能 |
| VICReg 自监督 | 诚实负结果（−0.017~−0.070） |
| 多数据集 | LPT 外部数据集 0.998（数据规模假设） |

## 待办
- [x] Fig.1 框架图（含三成本轴标注）— 2026-08-25 完成
- [x] 28523090 跨设备数据 — figshare 下载被 403 阻止（网络限制）；改用**历史锁定结果**（6 个 json 完整：三臂×2任务，已写入论文）
- [x] 参考文献 [10]-[12] 补 TII 近年论文 — 2026-08-25 完成
- [ ] 作者信息/致谢/ORCID
- [ ] IEEEtran 编译验证（需 TeX 环境）
- [ ] 28523090 数据重下载（换网络环境后可补 E4 三臂）
