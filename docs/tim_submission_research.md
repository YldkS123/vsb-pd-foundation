# IEEE TIM 投稿要求与评审标准调研报告

> 调研人：TIM-Expert（IEEE TIM 期刊专家）
> 日期：2026-08（基于 web_search 最新公开信息）
> 用途：团队 Paper 1《面向三相局部放电的覆盖感知采样与轻量层级弱监督检测》冲击 IEEE TIM 的投稿依据

---

## 1. 期刊范围（Scope）

**官方定位**：IEEE Transactions on Instrumentation and Measurement（TIM）是 IEEE 仪器与测量学会（IMS）的旗舰期刊，被公认为"通用仪器与测量领域排名第一的期刊"（IEEE I2MTC 官方材料原话：*"TIM is the number 1 journal in the area of general Instrumentation and Measurement"*）。

- 官方主页：https://ieee-ims.org/publication/ieee-tim
- 投稿须知（Information for Authors）：https://ieee-ims.org/publication/ieee-tim/information-authors
- 指标：JCR 影响因子近年稳定在 **5.5–6.0 区间**（2023 JCR ≈ 5.6），**中科院大类分区 2 区 Top**（JCR Q1；2025 版中科院分区为二区，见 [合肥工业大学论文标注](http://faculty.hfut.edu.cn/dingxu/en/lwcg/229252/content/14704.htm#lwcg)），属于仪器仪表方向顶级期刊（来源：[X-MOL 期刊页](https://www.x-mol.com/8paper/1201710354036232192)、[生知库介绍](https://www.biocloudy.com/news/detail/3088)）。

**覆盖方向**（对本文的匹配度判断）：
| TIM 关注方向 | 与本文匹配度 | 说明 |
|---|---|---|
| 测量理论与方法、测量科学 | ★★★★★ | 覆盖感知采样属于"测量策略/采样方法"创新，天然是 TIM 的菜 |
| 传感器与传感系统、信号采集前端 | ★★★★☆ | 本文若涉及三相 PD 信号采集/传感器布置，可强化此角度 |
| 信号处理与测量数据分析 | ★★★★★ | 层级弱监督检测是测量数据处理方法 |
| 机器学习在测量中的应用 | ★★★★☆ | TIM 大量接收 ML 论文，**但必须以测量/仪器为语境**，纯算法刷榜不受欢迎 |
| 电力设备在线测量与状态监测 | ★★★★★ | PD 检测/诊断是 TIM 高频主题（见第 4 节论文证据） |
| 计量学、不确定度分析 | ★★★☆☆ | 若能加入检测结果的统计可靠性/不确定度分析是加分项 |

**结论**：本文主题（三相 PD 覆盖感知采样 + 轻量层级弱监督检测）**高度契合 TIM 范围**，前提是叙事上把"测量"放在中心（采样策略→测量信息完整性/效率权衡；弱监督→测量标注成本问题的解法），而不是讲成一个纯深度学习分类故事。

---

## 2. 论文格式要求

**模板与版式**：
- 使用 IEEE 官方 **IEEEtran（LaTeX）模板**，双栏排版（US Letter，10pt）；Word 亦可，但 LaTeX 是主流且排版最稳。模板入口：[scispace TIM 模板](https://scispace.com/templates/ieee-transactions-on-instrumentation-and-measurement-template-ngnyzysnbqspl2q)、[typetex IEEEtran](https://www.typetex.app/templates/ieee/ieeetran)。
- 结构：Abstract → Index Terms → I. Introduction → II. ... → Conclusion → Acknowledgment → References；各节编号用罗马数字，小节用 A./B./C.。

**篇幅限制（重要）**：
- 常规论文（Regular Paper）**建议 8 页以内**（双栏排版后页数）；超过免费页数需签署并缴纳**超页版面费（Overlength Page Charge）**，一般允许到 10–12 页，超出过多会被退回压缩。
- 官方依据（IMS 官网的超页收费协议文件，2020/2024/2025 各版本均在更新）：
  - [Overlength Page Charge Agreement（2025-05 在线版）](https://ieee-ims.org/sites/ieeeims/files/2025-05/On-Line%20Overlength%20Page%20Charge%20Agreement_0.pdf)
  - [Overlength Page Charge Agreement（2024-09 版）](https://ieee-ims.org/sites/ieeeims/files/2024-09/Overlength%20Page%20Charge%20Agreement%20Form%20Updated%20September%202024.pdf)
  - [Overlength Agreement（2020-10 版）](https://ieee-ims.org/sites/ieeeims/files/2021-01/OverlengthAgreement%2010%202020.pdf)
- IEEE 标准超页费率约 **$210/页**（以协议文件当期为准）；常规录用无强制版面费，有自愿版面费传统（录用后编辑部会询问）。
- **投稿策略建议**：正文+图表压到 8–10 页；如果图表多，可接受 1–2 页超页费，但绝不要超过 12 页。

**摘要**：
- 约 **200–250 词**，一段式，必须点明：问题、方法、关键结果（量化指标）、意义。TIM 尤其看重摘要里体现"测量贡献"。

**图表规范**：
- 图表编号连续（Fig. 1…, TABLE I…），图题在图下方、表题在表上方，双栏内单栏宽约 3.5 in、跨栏约 7.16 in；矢量图（PDF/EPS），字体不小于 8pt；坐标轴标注物理量+单位。
- 公式用 IEEE 编号格式（(1), (2)…）；符号表可选。

**参考文献**：
- **IEEE 格式**（编号按正文出现顺序 [1][2]…，作者、标题、期刊缩写斜体、卷期页、年份、DOI）。完整样式示例见 [Paperpile 的 TIM 引用样式页](https://paperpile.com/s/ieee-transactions-on-instrumentation-and-measurement-citation-style/)。LaTeX 用 `IEEEtran`/`IEEEtranBST`。
- 建议引用 40–60 篇，覆盖：PD 检测综述与标准（IEC 60270/62478）、UHF/TEV/声学 PD 传感、深度学习 PD 模式识别、弱监督/半监督/多示例学习、轻量化网络、覆盖采样/信息论等——保证 TIM 审稿人熟悉的"测量圈"文献在列。

**其他硬性要求**：
- Index Terms：5–10 个（IEEE 标准）。
- 页脚/首页须含：作者姓名单位、资助声明（如有）、ORCID（可选）。
- 查重：投稿即走 Crossref/iThenticate 查重，重复率过高直接拒稿；禁止一稿多投。

---

## 3. 评审流程与标准

**流程**（ScholarOne Manuscripts）：
1. 投稿 → 编辑初审（Editorial Screening）：范围不符、格式明显不合规、查重超标 → **desk reject**（这类约占拒稿相当比例，格式务必一次到位）。
2. 分派副主编（Associate Editor）→ 送 **2–3 位（通常 3 位）审稿人**，单盲（审稿人可见作者信息）。
3. 一审周期：**常见 2–4 个月**；TIM 在 IEEE 事务期刊中以"处理利索、不拖泥带水"著称（[投稿体验分享](https://m.toutiao.com/article/7336764184476140059/)、[知乎投稿经历](https://zhuanlan.zhihu.com/p/648049296)）。
4. 决定：Major Revision / Minor Revision / Reject；修改后通常再审或直接接收；**从投稿到录用常见 4–8 个月**（案例：[2023 年投稿到见刊记录](https://zhuanlan.zhihu.com/p/660774094)、[CSDN 投稿记录](https://blog.csdn.net/weixin_47006934/article/details/129072195)）。
5. 录用后：IEEE 生产流程（PDF eXpress 排版校对）→ 在线发表。

**录用标准（TIM 审稿人通常看重的四件事，按权重排序）**：
1. **测量科学贡献（最重要）**：是否有新的测量原理/方法/传感器/测量系统/测量数据处理方法？采样策略、测量效率、信息完整性这类"测量方法论"创新正是 TIM 的核心兴趣点。
2. **实验严谨性**：真实采集数据（实验室或现场）而非纯仿真；与既有方法/标准方法的定量对比；足够的样本量与统计显著性；可复现性（数据、代码、参数公开是加分项）。
3. **实际应用价值**：面向真实工业场景（电力设备在线监测等），有现场验证或清晰的落地路径。
4. **新颖性与技术完备性**：相对已有工作的增量必须明确；方法章节推导/描述完整。

**常见拒稿原因**（社区经验，[知乎投稿指南](https://zhuanlan.zhihu.com/p/594070972)、[拒稿到接收经验帖](https://blog.csdn.net/weixin_28335283/article/details/158200569)）：
- 纯 ML/模式识别论文，无测量或仪器角度（"这是 ICLR 的稿子，不是 TIM 的"）；
- 实验只有仿真/公开数据集，缺乏真实测量数据验证；
- 与已发表工作增量过小；
- 缺乏误差/不确定度分析、缺乏与标准测量手段的对比；
- 写作不符合 IEEE 风格（引言无贡献列表、实验无消融、结论无工程意义）。

**对本文的直接启示**：
- 引言必须给出清晰的 **3–5 条贡献列表**；
- 实验必须有**真实三相 PD 数据**（实验室缺陷模型 + 如可能现场数据），并与 IEC 60270/UHF/TEV 等标准测量或经典检测方法对比；
- 弱监督部分要量化"标注成本节约"（如 5%/10% 标注下的性能 vs 全监督），这是 TIM 审稿人爱看的数据；
- 建议补充统计显著性检验（重复实验均值±方差）或不确定度讨论。

---

## 4. 近期 TIM 类似论文调研（叙事结构与侧重点）

### 4.1 硬件测量贡献型：GIS 外部 UHF PD 检测传感器
- **论文**：*External Partial Discharge Detection of Gas-Insulated Switchgears Using a Low-Noise and Enhanced-Sensitivity UHF Sensor Module*，IEEE TIM, 2023, DOI: 10.1109/TIM.2023.3277980（[IEEE Xplore](https://ieeexplore.ieee.org/document/10129983)、[NTU 机构库](https://dr.ntu.edu.sg/entities/publication/67ac6ac0-6038-48bd-bbe2-cbc831e9f2a5)）
- **叙事结构**：GIS PD 危害与检测需求 → 外部 UHF 检测的信噪比/灵敏度瓶颈 → 低噪声、高灵敏度 UHF 传感器模块设计（前端放大、屏蔽、耦合结构）→ 实验室 PD 试验平台验证（多种缺陷模型）→ 现场/工程验证。
- **侧重点**：测量硬件贡献（传感器+前端电路设计）是第一卖点；性能用灵敏度、信噪比、检测距离等**测量学指标**量化；ML 只是辅助分类。

### 4.2 测量数据处理型：生成式零样本 PD 诊断
- **论文**：*Generative Zero-Shot Learning for Partial Discharge Diagnosis in Gas-Insulated Switchgear*，IEEE TIM, 2023（[IEEE Xplore](https://ieeexplore.ieee.org/document/10098118)、[X-MOL](https://www.x-mol.com/paper/1646924245163061248?adv)、[Semantic Scholar](https://www.semanticscholar.org/paper/Generative-Zero-Shot-Learning-for-Partial-Discharge-Wang-Yan/a527e26010cd2401362c2a6bb5e18bdce6cc246b)）
- **叙事结构**：GIS PD 类型识别依赖大量标注样本（测量数据标注成本高）→ 生成式零样本学习框架（未见类特征生成 + 分类器）→ 在真实 PD 测量数据上验证。
- **侧重点**：**数据稀缺/标注成本**这一"测量数据科学"问题为动机；方法创新 + 真实数据实验；强调诊断在实际运维中的价值。与本文的"轻量层级弱监督检测"动机高度同源——**本文应引用并超越它**。

### 4.3 ML 替代测量型：PD 起始电压预测
- **论文**：*Prediction of Partial Discharge Inception Voltage for Electric Vehicle Motor Insulation Using Deep Learning*，IEEE TIM（[X-MOL](https://www.x-mol.com/paper/1657503943697453056?adv)、[南工大机构库](https://pure.njtech.edu.cn/en/publications/prediction-of-partial-discharge-inception-voltage-for-electric-ve/)）
- **叙事结构**：PDIV 是 EV 电机绝缘的关键测量指标但测量耗时昂贵 → 用 DL 从绝缘特征预测 PDIV（测量替代/加速）→ 实验验证预测精度。
- **侧重点**：ML 服务于"测量效率"——测量太贵/太慢，用模型替代。这与本文"覆盖感知采样降低测量/标注开销"的叙事同构。

### 4.4 补充参考（与弱监督/半监督最相关的 TIM 论文）
- *Operation Condition Assessment for Elevators Based on Deep Siamese Network and T-S Semi-Supervision Model*，IEEE TIM（[X-MOL](https://www.x-mol.com/paper/1765785819499171840?adv)）——半监督工业状态评估。
- *An Open-Set Semi-Supervised Contrastive Learning for Bearing Fault Diagnosis*（[哈工大学者页](https://scholar.hit.edu.cn/en/publications/an-open-set-semi-supervised-contrastive-learning-for-bearing-faul/)）——半监督+开集故障诊断，与本文"弱监督检测"方法线最接近，务必纳入对比与引用。

### 4.5 共性叙事模式总结（供 Paper-Reviser 参考）
TIM 测量+ML 论文的标准骨架：
1. **Introduction**：工业测量问题 → 现有测量/检测方法不足（性能/成本/标注）→ 本文方法一句话 → **贡献列表（3–5 条，编号）** → 组织结构；
2. **Method**：测量系统/采样策略 → 信号处理/特征 → 模型结构（公式+框图）→ 训练/推理细节；
3. **Experiments（篇幅最大，通常占 30–40%）**：实验平台与数据（真实采集，含设备型号、采样率、缺陷类型）→ 评价指标 → 与 SOTA/标准方法对比 → 消融 → 鲁棒性/噪声/标注比例实验 → 统计可靠性；
4. **Conclusion**：贡献回顾 + 工程意义 + 未来工作。
侧重点永远是：**真实测量数据、定量对比、工程可落地**。

---

## 5. TIM 与 Measurement、IEEE Sensors Journal 的定位差异

| 维度 | **IEEE TIM** | **Measurement (Elsevier)** | **IEEE Sensors Journal** |
|---|---|---|---|
| 主办/定位 | IEEE IMS 旗舰刊，测量科学为核心 | Elsevier，测量科学与技术综合刊 | IEEE Sensors Council，传感器技术刊 |
| 影响因子 | ≈ 5.6（1 区 Top） | ≈ 5.5（1 区） | ≈ 4.3（2 区） |
| 发文量 | 中等（~2000/年） | 很大（~6000+/年） | 极大（~4500+/年） |
| 一审周期 | 2–4 个月 | 常 1–2 个月（快） | 常 1–2 个月（快） |
| 深度/门槛 | 高：必须测量科学贡献 + 严谨实验 | 中：应用驱动即可，广度优先 | 中低：有传感器相关创新即可，量大 |
| 适合什么 | 新测量方法/传感器/测量数据处理、计量与不确定度、工业测量应用 | 测量应用、方法+应用结合、工程测量 | 传感器器件/接口/机理、传感应用 |
| 不适合什么 | 纯算法刷分、无测量语境的 ML | 深度不足或太"IEEE 风格"的理论 | 无传感器器件的纯信号处理/ML |

**什么类型的投稿容易中 TIM**（社区共识，[生知库分析](https://www.biocloudy.com/news/detail/3088)、[知乎指南](https://zhuanlan.zhihu.com/p/594070972)）：
- 有明确的**测量学创新点**（新测量原理/方法/指标/采样策略/不确定度处理）；
- 实验有**真实硬件/现场数据**，与标准方法（IEC 等）对比；
- ML 论文绑定测量语境：传感器数据、物理特征、采集成本、测量效率；
- 电力设备状态监测（PD、绝缘诊断、变压器/开关柜/GIS）、工业测量、生物医学测量等应用方向是传统强区。

**本文的定位建议**：走"测量策略（覆盖感知采样）+ 测量数据科学（层级弱监督）"双线，主打**采样-标注-检测联合优化降低测量成本**的故事——这比单纯"又一个 PD 分类网络"更符合 TIM 口味。

---

## 6. 投稿系统与流程

- **投稿系统**：ScholarOne Manuscripts（TIM 使用，入口 https://mc.manuscriptcentral.com/tim-ieee ；首次需注册 ORCID）。
- **投稿材料**：主稿 PDF（或 LaTeX 源）、图/表文件、**Cover Letter**、作者简介（IEEE 格式）、可选补充材料。
- **Cover Letter 要点**：
  1. 论文题目 + 一段式贡献总结（新颖性、与现有工作的区别）；
  2. 说明与 TIM 范围的契合（测量贡献点）；
  3. 声明：未一稿多投、无利益冲突、全体作者同意；
  4. 如适用，说明与作者已发表工作的区别（避免自我抄袭质疑）。
- **开放获取**：TIM 是**混合期刊**（Hybrid）——可自选传统订阅模式（无强制 APC，录用后有自愿版面费/超页费传统）或 **IEEE Open Access**（APC 数千美元级，具体以 IEEE Open 官网当期费率为准；IMS 会员有折扣）。
- **时间线预期**：投稿 → 编辑初审（1–3 周）→ 一审（2–4 个月）→ 修改（1–2 个月）→ 录用 → 在线发表。全程 4–8 个月是常态，规划投稿时间需预留。

---

## 7. 给团队的落地清单（本报告的可执行结论）

1. **定位**：标题/摘要/引言突出"测量"——覆盖感知采样 = 测量策略创新；层级弱监督 = 测量标注成本解法。避免纯 ML 叙事。
2. **篇幅**：IEEEtran 双栏，正文+图表目标 8–10 页，绝不超 12；摘要 ≤200–250 词；Index Terms 5–10 个。
3. **实验**：必须有真实三相 PD 数据（缺陷模型+采样装置），与标准方法对比（IEC 60270/UHF/TEV 或经典检测器），加弱监督标注比例实验（如 5%/10%/20% 标注 vs 全监督）+ 统计可靠性（均值±方差/显著性）。
4. **文献**：引用第 4 节论文（尤其 4.2 生成式零样本 PD 诊断、4.4 两篇半监督诊断）作为对标，并在对比实验中覆盖同类弱监督/半监督方法。
5. **写作**：IEEE 风格——贡献列表、方法公式化、实验定量、结论工程意义；Cover Letter 按第 6 节要点撰写。
6. **投稿**：ScholarOne 提交；格式一次性合规（避免 desk reject）；按时间线倒排计划。

---

## 主要信息来源（链接）

- [TIM 官方主页（IEEE IMS）](https://ieee-ims.org/publication/ieee-tim)
- [TIM Information for Authors（官方）](https://ieee-ims.org/publication/ieee-tim/information-authors)
- [TIM 超页收费协议 2025-05](https://ieee-ims.org/sites/ieeeims/files/2025-05/On-Line%20Overlength%20Page%20Charge%20Agreement_0.pdf)、[2024-09 版](https://ieee-ims.org/sites/ieeeims/files/2024-09/Overlength%20Page%20Charge%20Agreement%20Form%20Updated%20September%202024.pdf)
- [TIM 期刊页（X-MOL，IF 与指标）](https://www.x-mol.com/8paper/1201710354036232192)
- [TIM 投稿指南（知乎）](https://zhuanlan.zhihu.com/p/594070972)、[投稿经历（知乎）](https://zhuanlan.zhihu.com/p/648049296)、[2023 投稿到见刊记录（知乎）](https://zhuanlan.zhihu.com/p/660774094)
- [TIM 投稿记录（CSDN）](https://blog.csdn.net/weixin_47006934/article/details/129072195)、[拒稿到接收经验（CSDN）](https://blog.csdn.net/weixin_28335283/article/details/158200569)
- [TIM 期刊收录偏好与通关技巧（生知库）](https://www.biocloudy.com/news/detail/3088)
- [TIM 审稿体验（今日头条）](https://m.toutiao.com/article/7336764184476140059/)
- [TIM 引用格式（Paperpile）](https://paperpile.com/s/ieee-transactions-on-instrumentation-and-measurement-citation-style/)
- [IEEEtran 官方 LaTeX 模板（typetex）](https://www.typetex.app/templates/ieee/ieeetran)
- 对标论文：[GIS 外部 UHF PD 检测（TIM 2023）](https://ieeexplore.ieee.org/document/10129983)、[生成式零样本 PD 诊断（TIM 2023）](https://ieeexplore.ieee.org/document/10098118)、[EV 电机 PDIV 预测（TIM）](https://www.x-mol.com/paper/1657503943697453056?adv)、[电梯半监督状态评估（TIM）](https://www.x-mol.com/paper/1765785819499171840?adv)、[开集半监督对比学习轴承诊断](https://scholar.hit.edu.cn/en/publications/an-open-set-semi-supervised-contrastive-learning-for-bearing-faul/)
