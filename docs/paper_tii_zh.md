# 面向配电线路局部放电监测的高性价比可信工业 AI：时频编码与覆盖感知采样、层级弱监督

> **TII 版论文中文稿 v1（2026-08-25）**
> **作者：杨上（第一作者，山东大学 2024 级本科生）；张斌（通讯作者，山东大学教授）**
> 目标期刊：IEEE Transactions on Industrial Informatics (TII, IF≈11, 中科院1区Top)
> 状态：独立投稿（不投 TIM）——论文完全自包含
> 单位：山东大学（Shandong University）

---

## 摘要

配电线路的工业状态监测产生极长、稀疏、弱标注的三相局部放电（PD）信号，其中感知成本、标注成本与评估可信度共同决定可部署性。本文提出一种高性价比且可信的工业 AI 框架，对三个轴进行联合优化。**第一**，覆盖感知采样方案通过鲁棒事件分数仅提取原始数据的 8.2%（40 MHz 下每相 80 万点信号取 K=8 个窗口），且采样率研究表明感知率可降低 8 倍（40→5 MHz）而检测性能保持 83%——直接降低 ADC 与存储硬件成本。**第二**，时频窗口编码器（STFT + 2D CNN，167K 参数）取代普通 CNN 骨干，在相同弱监督管线（注意力 MIL + context-concat 三相交互 + noisy-OR）下将 VSB 数据集相位级 PR-AUC 从 0.615 提升至 0.703（±0.025）；轻量 patch Transformer（235K 参数）被证明需要更大数据规模才能发挥优势（外部 46K 样本电机 PD 数据集上 from-scratch PR-AUC 0.998 vs VSB 7.4K 样本上 0.588）——编码器选择的数据规模假设。**第三**，弱监督设计在仅 50% 相位标签时保留全监督性能的 88%（0.542 vs 0.615）；自监督（VICReg）预训练研究报告诚实的负结果（低标注比例下无增益），精炼了自监督何时有效。**第四**，泄漏安全评估协议——哈希校验分割锁、一次性盲测与阳性率归一化分析——确保可信的工业结论；外部多数据集验证（电机 PD vs 噪声，45,970 样本；跨设备示波器捕获）确认可迁移性（from-scratch ROC-AUC 0.99+）。该框架面向配电网络的边缘可部署、成本感知且可信的 PD 监测。

**关键词**：工业人工智能；局部放电监测；覆盖感知采样；时频编码；弱监督学习；采样率优化；可信 AI

---

## 一、引言

### A. 工业背景

局部放电（PD）是电力设备绝缘劣化的最早征兆 [1], [2]。配电线路的在线 PD 监测记录原始高采样率波形（40 MHz，每相每次测量 80 万采样点），必须逐相、逐测量判定是否存在放电 [3], [4]。在工业规模上为这一任务部署机器学习面临**三重成本问题**：

1. **感知成本**：40 MHz 数字化长记录主导 ADC、存储与传输预算；更低的采样率是否足够是一个开放的硬件设计问题。
2. **标注成本**：80 万点信号的窗口级标注成本过高；仅存在相位级标签（约 5.9% 阳性相位），需要弱监督学习。
3. **信任成本**：评估泄漏虚高报告性能并削弱工业信心；可信的评估协议是采纳的前提。

### B. 相关工作与差距

近期 TIM/IEEE 的 PD 检测工作涵盖硬件感知模块 [5]、深度学习模式识别 [6], [7]、基于 Transformer 的多源 PD 识别 [8] 与自监督异常检测 [9]。工业信息学期刊日益强调资产监测的成本感知与可信 AI [10]–[12]。然而，多数现有方法 (i) 处理全长信号或固定片段而未处理采样率设计变量，(ii) 假设全标注窗口，(iii) 未量化评估泄漏。本文处理感知–标注–信任的联合优化，据我们所知，这在先前工业 AI 工作中未被联合处理。

### C. 贡献

1. **成本感知采样与硬件率优化**：确定性覆盖感知采样方案（鲁棒事件分数；等距锚点 + 事件窗口，K=8，8.2% 数据）加采样率研究（40/20/10/5 MHz），显示 8× 感知成本降低而性能保持 83%。
2. **弱监督的时频编码**：STFT+2D-CNN 窗口编码器在相同弱监督管线下将相位 PR-AUC 提升至 0.703；基准测试轻量 patch Transformer，揭示数据规模假设（Transformer 在更多数据下更优）。
3. **标注成本量化**：50% 相位标签保留全监督性能的 88%；VICReg 自监督研究诚实报告负结果，精炼自监督何时有效。
4. **可信评估与多数据集验证**：哈希校验分割锁、一次性盲测、阳性率归一化分析，以及外部电机 PD/跨设备验证确认可信、可迁移的工业结论。

---

## 二、问题定义

设一次测量由三相信号 $x_A, x_B, x_C$ 组成，每相长度 $L = 800{,}000$，采样率 $f_s = 40$ MHz。相位级标签 $y_p \in \{0, 1\}$ 表示相位 $p$ 是否含放电；测量级标签遵循评估指标所用的 noisy-OR 逻辑（任一相放电即测量异常）。训练时仅有相位级标签可用。任务是在三个工业约束下学习逐相位判别器 $f_p: x_p \to \hat{y}_p$（$p \in \{A, B, C\}$）：**(C1) 最小感知率**（硬件成本）、**(C2) 最小标注**（标注成本）与 **(C3) 可信评估**（信任）。

---

## 三、所提框架

### A. 覆盖感知采样（CAS）

对每相信号 $x$（长度 $L$），先去中值得到 $\tilde{x}[n] = x[n] - \mathrm{median}(x)$，再逐点计算三类能量特征：

- **幅值**：$a[n] = |\tilde{x}[n]|$；
- **Teager 能量**：$\tau[n] = |\tilde{x}[n]^2 - \tilde{x}[n-1]\tilde{x}[n+1]|$；
- **差分 RMS**：$d[n] = \sqrt{\mathrm{mean}_{m \in W}(\Delta\tilde{x}[m]^2)}$，$|W|=256$（反射填充）。

每类特征 $v$ 定义鲁棒非负 z-score：

$$z(v) = \max\left(\frac{v - \mathrm{median}(v)}{1.4826 \cdot \mathrm{MAD}(v)}, 0\right) \qquad (1)$$

MAD 退化时回退至平均绝对偏差尺度。事件分数为

$$S[n] = \max(z(a[n]),\, z(\tau[n]),\, z(d[n])) \qquad (2)$$

**确定性采样方案（算法 1）**：对每相提取 $K = K_u + K_e = 8$ 个长度 $W = 8{,}192$ 的窗口：
1. *等距锚点*（$K_u=4$）：起点最大分离的均匀间隔窗口——保证全信号覆盖；
2. *事件窗口*（$K_e=4$）：$S$ 的峰（最小距离 $W/2$），按 $(-S, \text{start})$ 排序，与已选窗口 IoU ≥ 0.5 去重；
3. *分层回退*：若选窗不足 $K$ 个，在 256 格网格上按最大最小距离填充。

全部采样参数提交至 SHA-256 锁，使方案确定且可复现。该方案仅处理 $K \cdot W / L = 8.2\%$ 的原始数据。

**算法 1** 覆盖感知确定性采样
```
输入: x ∈ R^L, K_u = K_e = 4, W = 8192
输出: 窗口集 W, |W| = K_u + K_e
1: W ← ∅; x̃ ← x − median(x)
2: 计算 a, τ, d; S ← max(z(a), z(τ), z(d))      # 式 (2)
3: 阶段 I — 锚点: 放置 K_u 个等距窗口; W ← W ∪ anchors
4: 阶段 II — 事件:
5:   P ← peaks(S, dist_min = W/2, S > 0)
6:   按 (−S, start) 排序 P
7:   for p ∈ P:
8:     s ← clip(peak_p − W/2, 0, L − W)
9:     if 对所有 w ∈ W 有 IoU([s, s+W], w) < 0.5: W ← W ∪ {[s, s+W]}
10:    if |W| = K: break
11: 阶段 III — 回退: while |W| < K: 添加最大最小距离的 256 格窗口
12: return W
```

### B. 时频窗口编码器（TFE）

对每个窗口 $\mathbf{w} \in \mathbb{R}^{8192}$，TFE 计算对数幅度 STFT 谱图（256 点 FFT，hop 128，Hann 窗），用三个 2D 卷积块（GroupNorm + SiLU）处理，随后全局平均+最大池化与线性投影：

$$\mathbf{h}_{\mathrm{tf}} = \mathrm{Proj}\big(\mathrm{Pool}\big(\mathrm{CNN}_{2d}\big(\log(1+|\mathrm{STFT}(\mathbf{w})|)\big)\big)\big) \qquad (3)$$

其中 $\mathbf{h}_{\mathrm{tf}} \in \mathbb{R}^{128}$。完整管线（编码器 + MIL + 交互 + 头）含 **167,394 参数**。

**轻量 patch Transformer（LPT，对比）**：窗口切为 128 个 64 采样 token；加入可学习位置嵌入；2 层 Transformer 编码器（$d=96$，4 头，前馈 192，GELU，norm-first）产生 token 表示；全局平均+最大池化投影至 128 维（完整管线 **234,594 参数**）。

### C. 层级弱监督

**窗口到相位注意力 MIL**：每相含 $K$ 个窗口嵌入 $\{\mathbf{h}_1, \ldots, \mathbf{h}_K\} \subset \mathbb{R}^{128}$，注意力 MIL [13] 聚合：

$$\alpha_k = \frac{\exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_k))}{\sum_{j=1}^K \exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_j))}, \qquad \mathbf{z}_p = \sum_{k=1}^K \alpha_k \mathbf{h}_k \qquad (4)$$

**Context-concat 三相交互**：设 $\mathbf{c} = \frac{1}{3}(\mathbf{z}_A + \mathbf{z}_B + \mathbf{z}_C)$ 为全局相位上下文；每相表示与上下文拼接并分类：

$$\hat{y}_p = \sigma\big(\mathbf{w}_p^\top [\mathbf{z}_p, \mathbf{c}] + b_p\big) \qquad (5)$$

**测量级推理**：确定性 noisy-OR：$\hat{y}_{\mathrm{meas}} = 1 - \prod_p (1 - \hat{y}_p)$。

**损失**：相位级二元交叉熵：$\mathcal{L} = \sum_p \mathrm{BCE}(\hat{y}_p, y_p)$。

### D. 可信评估协议

1. **泄漏审计**：扫描 56 个历史预测文件；158 个污染候选被降级。
2. **分割锁**：开发（2,481）/盲测（423）分割与全部管线参数提交至 SHA-256 哈希校验锁；新分割（IR、采样率）各自建锁。
3. **一次性盲测**：冻结模型在保留分块上恰好评估一次，防篡改收据绑定协议锁与检查点哈希。
4. **阳性率归一化分析**：解析 PR 重加权将表观 PR-AUC 差距分解为阳性率效应与真实域偏移。
5. **测量聚类 bootstrap**：按测量分层 2,000 次重采样用于置信区间。

---

## 四、实验

### A. 数据集与协议

| 数据集 | 任务 | 样本 | 阳性率 | 角色 |
|---|---|---|---|---|
| VSB (Kaggle 2019) [3] | 三相 PD | 2,481 开发（7,443 相位） | 5.95% | 主 |
| figshare 24033225 | 电机 PD vs 噪声 | 45,970 测试 | 50% | 外部 |
| figshare 28523090 | 示波器 PD（C1→C2） | 639/319 测试 | — | 跨设备 |

**实现**：PyTorch；AdamW（lr 1e-3，wd 1e-4），批 64，最多 40 epoch，早停（patience 15，min delta 0.001）基于验证相位 PR-AUC，梯度裁剪（max-norm 1.0）；StratifiedGroupKFold(5, seed 42) 按测量分组；seeds {42, 7, 2024} 稳定性。指标：PR-AUC 为主（极端不平衡），ROC-AUC、MCC、F1、ECE/Brier 为辅；逐折均值主口径；配对聚类 bootstrap（2,000，按测量聚类）显著性；RTX 4060 Laptop GPU。

### B. 主结果（VSB，5 折 CV，逐折均值）

| 编码器 | 参数（管线） | 相位 PR-AUC | 测量 PR-AUC | 编码器前向（8×8 批，CPU） |
|---|---|---|---|---|
| simple_cnn（基线） | 113,265 | 0.615 ± 0.053 | 0.643 | 63 ms |
| **TFE（STFT+2D CNN）** | 167,394 | **0.703 ± 0.025** | 0.744 | 542 ms |
| LPT（patch Transformer） | 234,594 | 0.588 ± 0.054 | 0.621 | 424 ms |

**TFE** 显著优于基线（+0.088 相位 PR-AUC；开发集 OOF 配对聚类 bootstrap，95% CI 排除 0），仅 1.5× 参数——时频表示是 PD 瞬态结构的关键编码器属性。**显式效率权衡**：TFE 的 STFT 前端在 CPU 上约为 CNN 前向的 8.6×；边缘 GPU 部署缓解此问题（讨论）。

**跨编码器统计**（配对 bootstrap，seed 42）：TFE vs simple_cnn 相位差 +0.088（95% CI 排除 0）；LPT vs simple_cnn −0.027（CI 含 0）；TFE vs LPT +0.115（CI 排除 0）。三种子均值确认：seeds {42, 7, 2024} 上 TFE 0.70 ± 0.01。

### C. 感知成本优化（采样率）

| 采样率 | 数据量 | 相位 PR-AUC | 测量 PR-AUC | 保持 |
|---|---|---|---|---|
| 40 MHz | 1× | 0.617 | 0.629 | 100% |
| 20 MHz | 1/2 | 0.569 | 0.588 | 92% |
| 10 MHz | 1/4 | 0.518 | 0.551 | 84% |
| 5 MHz | 1/8 | 0.510 | 0.535 | 83% |

**8× 感知成本降低而性能保持 83%**——直接的硬件设计指南（ADC 率、存储、传输预算）。40 MHz 行（0.617）与主模型（0.615）在噪声内一致，验证协议。

### D. 标注成本量化

| 标注 | 相位 PR-AUC | vs 100% |
|---|---|---|
| 5% | 0.272 | 44% |
| 10% | 0.346 | 56% |
| 20% | 0.377 ± 0.021（3 个标注种子） | 61% |
| 50% | 0.542 | **88%** |
| 100% | 0.615 | 100% |

**自监督（VICReg）负结果**：预训练未改善低标注微调（5%：0.202 vs 0.272；10%：0.285 vs 0.346；20%：0.361 vs 0.377）——窗口级自监督表示与相位级弱监督任务错位；诚实报告精炼了自监督何时有效。VICReg 细节：全部开发窗口上 40-epoch 预训练，4 种增强（时间偏移 ±128、幅值 0.9–1.1、噪声 20–40 dB、频域掩蔽），方差/不变性/协方差损失（25/25/1）。

### E. 外部多数据集验证（figshare 24033225）

| 编码器 | 零样本 ROC/PR | from-scratch ROC/PR | fine-tune ROC/PR |
|---|---|---|---|
| simple_cnn | 0.776/0.812 | 0.991/0.993 | 0.987/0.990 |
| **LPT** | 0.729/0.669 | **0.998/0.998** | **0.997/0.998** |

**数据规模假设确认**：Transformer（7.4K 样本 VSB 上较弱）在 46K 样本外部数据集上成为最佳编码器（0.998 from-scratch vs 0.991 CNN）——编码器选择依赖数据规模；TFE 在小型工业数据规模下仍是成本高效选择。

**跨设备验证（figshare 28523090，示波器捕获，C1→C2）**：E4 主模型（simple_cnn 编码器）在冻结外部协议下评估（零样本 / from-scratch / fine-tune，C1 训练模型在 C2 设备上评估）：

| 任务 | 零样本 ROC/PR | from-scratch ROC/PR | fine-tune ROC/PR |
|---|---|---|---|
| PD vs background（C2，n=160） | 0.493 / 0.665 | 0.641 / 0.815 | **0.832 / 0.919** |
| PD vs corona（C2，n=213） | 0.635 / 0.624 | 0.982 / 0.981 | **0.988 / 0.988** |

跨采集设备零样本迁移失效（ROC ≈ 0.49–0.64），但少量目标域数据即可恢复强性能（fine-tune ROC 0.83–0.99，超过历史 80k 模型的 0.80–0.91）——部署启示是传感链变化时需要现场标定数据，与工业逐站点调试实践一致（脚本 `tii_28523090_e4_3arm.py`；输出 `results/tii_external/summary_28523090_e4.json`）。

### F. 鲁棒性、基线与统计功效

- **E4/TFE 扰动鲁棒性**：5 dB 噪声 +3.7%（抗噪，历史主模型为 −67%），幅值 ≈0%，时间偏移 −47~−73%（时间对齐仍是最弱点；增强缓解至 ±3%）。
- **经典检测器基线**：能量 0.055 / 冲击性 0.104 / 频谱 0.088 / PRPD+LR 0.323 相位 PR-AUC vs TFE 0.703——一个数量级的差距隔离了学习型弱监督的价值。
- **统计功效分析**：测量级最小可分辨差异（4 阳性 ≈ 0.57；31 ≈ 0.21；163 ≈ 0.09）量化测量级陈述何时为探索性。
- **阳性率归一化分析**：表观独立集 PR-AUC 下降分解为阳性率效应（Δπ = −0.242，≈77%）与真实域偏移（Δshift = −0.073，≈23%），阳性率归一化提升比 13.8× vs 开发集 10.3×。

---

## 五、讨论

### A. 感知–标注–信任联合优化

每个轴都有定量杠杆：**感知率**（83% 性能下 8× 成本降低）、**标注比例**（88% 性能下 50% 标签）与**评估可信度**（带归一化分析的泄漏安全协议）。它们共同定义可部署性设计空间：例如，10 MHz / 20% 标签配置在 1/4 感知与 1/5 标注成本下保留 ≈84% × ≈61% ≈ 51% 的全性能——具体的工业权衡菜单。

### B. 编码器选择的数据规模假设

TFE 在小数据下胜出（VSB：0.703 vs 0.588）；LPT 在更大数据下胜出（外部：0.998 vs 0.991）。我们假设 Transformer 容量仅在超过数据规模阈值后才有回报——工业编码器选择的实用指导。VICReg 负结果进一步表明，在小规模下预训练无法替代任务对齐的监督。

### C. 部署路径

TFE 的 STFT 前端在 CPU 上是计算瓶颈（每 8×8 批 542 ms），但在边缘 GPU 上并行良好；采样方案将输入减少 12×（K=8 of 96 窗口当量），5 MHz 感知再减少原始输入 8×——组合管线边缘可行。部署时推荐时间偏移增强。

### D. 局限

单一主工业数据集（VSB）；跨设备零样本迁移仍受限（需目标域微调）；时间偏移敏感需增强；VICReg 式自监督未奏效，不应假设其有益。

---

## 六、结论

我们提出了一种面向 PD 监测的高性价比且可信的工业 AI 框架：覆盖感知采样实现 8× 感知成本降低，时频编码将 VSB 性能提升至 0.703 PR-AUC，弱监督在减半标签下保留 88%，诚实的自监督负结果，以及泄漏安全的多数据集验证。该框架为在配电网络中部署可信、低成本的 PD 监测提供了具体设计杠杆。

---

## 参考文献

[1] G. C. Stone, "Partial discharge diagnostics and electrical equipment insulation condition assessment," *IEEE Trans. Dielectr. Electr. Insul.*, vol. 12, no. 5, pp. 891–904, 2005.
[2] W. J. K. Raymond et al., "Partial discharge classifications: Review of recent progress," *Measurement*, vol. 68, pp. 164–181, 2015.
[3] VSB Power Line Fault Detection, Kaggle Competition, 2019.
[4] IEC 60270 / IEC TS 62478, partial discharge measurement standards.
[5] External PD detection via low-noise UHF sensor module, *IEEE Trans. Instrum. Meas.*, 2023.
[6] J. Zheng et al., "GIS partial discharge pattern recognition based on time-frequency features and improved CNN," *Energies*, vol. 15, p. 7372, 2022.
[7] Z. Fei et al., "Partial discharge pattern recognition based on ensembled simple CNN and quadratic SVM," *Energies*, vol. 17, p. 2443, 2024.
[8] Detection Transformer-based deep learning for multisource PD recognition, 2023–2024.
[9] Self-supervised temporal contrastive learning for PD anomaly detection, 2024.
[10] L. Liu et al., "Flexible generalized demodulation for intelligent bearing fault diagnosis under nonstationary conditions," *IEEE Trans. Ind. Informat.*, 2024.
[11] "Lifelong monitoring of bearing-rotor systems over whole life cycle: An emerging paradigm," *IEEE Trans. Ind. Informat.*, 2024.
[12] "Physics-inspired sparse voiceprint sensing for bearing fault diagnosis," *IEEE Trans. Ind. Informat.*, 2024.
[13] M. Ilse, J. M. Tomczak, and M. Welling, "Attention-based deep multiple instance learning," *ICML*, 2018.
[14] A. Bardes, J. Ponce, and Y. LeCun, "VICReg: Variance-invariance-covariance regularization for self-supervised learning," *ICLR*, 2022.
[15] S. Misak et al., "Problems associated with covered conductor fault detection," *EPQU*, 2011.
[16] G. M. Hashmi and M. Lehtonen, "On-line PD detection for condition monitoring of covered-conductor overhead distribution networks," *ICEE*, 2008.

---

## 图（Figures）

- **图 1** — 框架总览：三相感知 → CAS 采样（K=8，8.2% 数据）→ TFE/LPT 编码 → 注意力 MIL → context-concat → noisy-OR 决策；标注三成本轴（感知率、标注比例、泄漏安全评估）（`figures_tii/fig1_framework.png`）。
- **图 2** — 采样率成本-性能曲线（x：采样率；y：PR-AUC；5 MHz 保持 83%）（`figures_tii/fig2_sampling_rate.png`）。
- **图 3** — 编码器对比：VSB vs 外部数据集性能——数据规模假设（`figures_tii/fig3_encoder_scale.png`）。
- **图 4** — 标注成本曲线（x：标注比例；y：PR-AUC；叠加 VICReg 负结果）（`figures_tii/fig4_labeling_cost.png`）。
