# NCAA 2026 竞赛需求与 goto_conversion 数据指南

本文档回答三个核心问题：
1. **Kaggle 要求预测什么？**（预测对象与提交格式）
2. **NCAA 锦标赛赛制是什么？**（赛制与对阵结构）
3. **goto_conversion Substack 提供什么数据？应下载哪些？如何用于提交？**

---

## 一、Kaggle 竞赛要求预测什么

### 1.1 预测对象

**不是**「实际会发生的每一场比赛」，而是 **所有可能的「两队对阵」组合**。

- 男子（Men）：约 68 支参赛队 → 两两配对约 **C(68,2) ≈ 2,278** 对
- 女子（Women）：约 68 支参赛队 → 同理约 **2,278** 对
- **Stage2 共需预测约 132,131 个对阵**（来自 `SampleSubmissionStage2.csv` 行数）

这些对阵中，只有很少一部分会真正发生（淘汰赛单败，每轮只打一半场次）。但 Kaggle 要求对「所有可能的」对阵都给出预测，因为真实结果公布后，会按 Brier Score 对所有对阵的预测和实际 0/1 结果计算平均得分。

### 1.2 提交格式

| 字段 | 格式 | 说明 |
|------|------|------|
| **ID** | `SSSS_XXXX_YYYY` | 赛季(4位) + 小 TeamID(4位) + 大 TeamID(4位)，如 `2026_1101_1102` |
| **Pred** | [0, 1] 之间实数 | **第一个队（XXXX）** 的获胜概率；第二个队胜率 = 1 - Pred |

示例：
```
ID,Pred
2026_1101_1102,0.5
2026_1101_1103,0.5
...
```

- ID 中 **XXXX < YYYY**（按 TeamID 升序排列）
- Pred 表示 **XXXX 这支球队** 击败 YYYY 的概率

### 1.3 TeamID 约定

- **男子**：TeamID 范围约 1000–1999
- **女子**：TeamID 范围约 3000–3999
- 队名与 TeamID 的映射见：`MTeamSpellings.csv`、`WTeamSpellings.csv`

### 1.4 评分方式：Brier Score

```
Brier = mean((Pred - actual)^2)
```

- `actual = 1`：该队赢
- `actual = 0`：该队输
- **分数越低越好**

---

## 二、NCAA 锦标赛赛制（简要）

### 2.1 规模与阶段

- **68 队参赛**（男子、女子各 68 队）
- **Selection Sunday**：2026 年 3 月 15 日（DayNum=132），公布入围队伍与种子
- **Play-in（First Four）**：DayNum 134–135，4 场附加赛，68 → 64 队
- **正赛 6 轮**（64 队单败淘汰）：
  - Round 1（DayNum 136–137）：64 → 32
  - Round 2（DayNum 138–139）：32 → 16（Sweet Sixteen 前）
  - Round 3（DayNum 143–144）：16 → 8（Sweet Sixteen）
  - Round 4（DayNum 145–146）：8 → 4（Elite Eight）
  - Round 5（DayNum 152）：4 → 2（Final Four）
  - Round 6（DayNum 154）：2 → 1（冠军）

### 2.2  bracket 结构

- 四个分区：W、X、Y、Z
- 种子 1–16，同区内 1v16、2v15… 等首轮对阵固定
- 不同区球队最早在 Final Four（Round 5/6）才可能相遇

因此，「哪些两队可能相遇、在哪一轮相遇」由 bracket 和种子决定。`get_roundOfMatch(team1, team2, seeds_df)` 等函数就是基于此推断对阵轮次。

---

## 三、goto_conversion Substack 提供什么

### 3.1 数据来源与形式

- **网站**：[goto_conversion Substack](https://gotoconversion.substack.com/archive)
- **内容**：2026 年男女篮 **每周** 更新的 projection table
- **格式**：每篇文章底部提供 **XLSX 文件下载**

示例文章：
- [Mens March Madness 2026 Weekly Projection (23 Feb)](https://gotoconversion.substack.com/p/mens-march-madness-2026-weekly-projection-bf9)
- 底部有：`Mensmarchmadness 20260222 235109.xlsx` 下载链接

### 3.2 Projection Table 内容（与 goto_conversion 输出一致）

根据项目内 `goto_conversion/outputFiles/mensProbabilitiesTable2025.csv` 的结构，projection table 大致为：

| 列名 | 含义 |
|------|------|
| `player` | 队伍名称（或占位符如 Y16、W16、X11） |
| `Elo_Rating` | Elo 评分（可选） |
| `rd1_win` | 晋级/赢得第 1 轮的概率 |
| `rd2_win` | 晋级第 2 轮的概率 |
| … | … |
| `rd6_win` | 夺得冠军的概率 |

- 每行对应一支队伍（或 bracket 中的一个 slot）
- 表中行序与 bracket 结构相关，相邻行常为同区、同轮潜在对手

### 3.3 Substack 与完整算法的差异

Substack 文章中有说明：
> "For efficiency purposes, this article uses an **approximation algorithm** to compute the projection table. The **full algorithm**, which computes a sharper projection table, will be used **after Selection Sunday**."

- **现阶段（2 月–3 月 15 日前）**：使用近似算法，计算更快
- **Selection Sunday 之后**：使用完整算法，强弱差距会更明显
- **建议**：正式提交前，优先使用 **3 月 15 日之后** 的 projection table

### 3.4 需要下载的 Substack 内容

| 用途 | 建议下载 |
|------|----------|
| 本地开发 / 验证 | 最新一期男/女 projection table（各一个 XLSX） |
| 正式 Kaggle 提交 | **Selection Sunday（2026-03-15）之后** 发布的最新 projection table（男女各一） |

**下载步骤**：
1. 打开对应 Substack 文章（如 "Mens March Madness 2026 Weekly Projection (7 Mar)"）
2. 滚动到文章底部
3. 点击 XLSX 旁的 **Download** 下载

---

## 四、如何将 Substack 数据转为 Kaggle 提交

### 4.1 数据流概览

```
Substack XLSX (projection table)
    ↓ 解析为 player, rd1_win..rd6_win
    ↓ 通过 MTeamSpellings/WTeamSpellings 映射 player → TeamID
    ↓ 根据 bracket 与 rd*_win 推导 pairwise P(A beats B)
SampleSubmissionStage2.csv 格式的 submission
```

### 4.2 关键难点

1. **Projection table 是「晋级各轮概率」**，不是直接的 pairwise 胜率
2. **需要 bracket 信息**（种子、分区）才能知道「哪两队可能在哪轮相遇」
3. **goto_conversion 完整流程**：  
   赔率 → goto_conversion 算法 → 晋级概率表 →（结合 bracket）→ pairwise 概率矩阵 → submission

### 4.3 现有项目中的实现方式

- **ncaa2025-3rd-solution**、**updated-goto-conversion**：使用 Kaggle 上的 **538data** 数据集
  - 内含：`mensProbabilitiesTable.csv`、`womensProbabilitiesTable.csv`、以及部分方案中的 `submission.csv`
  - 这些是别人用 goto_conversion 产出并上传的
- **2026 赛季**：Kaggle 目前没有 2026 的 538data，需要：
  - 从 Substack 下载 2026 projection table，或
  - 自行运行 goto_conversion（若你能获取 538 等赔率源）生成概率表

### 4.4 推荐实施路径

1. **下载 Substack 最新 projection table（男、女各一）**  
   - 先用于本地开发和流程打通

2. **建立 player → TeamID 映射**  
   - 使用 `MTeamSpellings.csv`、`WTeamSpellings.csv`  
   - Substack 中的队名/占位符（如 Y16）需与 Kaggle 命名或 seed 对应

3. **实现「晋级概率 → pairwise 胜率」的转换逻辑**  
   - 参考 goto_conversion 的 [probabilityMatrices](https://github.com/gotoConversion/goto_conversion/tree/main/probabilityMatrices) 或相关文档  
   - 或参考 538data 中 pre-computed 的 submission 结构

4. **生成 submission**  
   - 对 `SampleSubmissionStage2.csv` 中的每个 ID，用 (team1, team2) 查 pairwise 胜率  
   - 若查不到（如某队不在 projection 中），可用默认值 0.5 或与 Raddar 等其他模型 blend

5. **应用 set_optimalStrategy 等后处理**  
   - 对接近 33.3% 胜率的「风险队」做策略性调整（见深度分析报告）

### 4.5 Selection Sunday 后的注意事项

- 2026-03-15 之后，**WNCAATourneySeeds.csv**、**MNCAATourneySeeds.csv** 会更新为 2026 实际种子
- 需用 **最新种子 + 最新 Substack projection table** 生成最终 submission
- Substack 在 Selection Sunday 后也可能发布使用「完整算法」的 projection，质量更高

---

## 五、总结对照表

| 问题 | 答案 |
|------|------|
| 要预测什么？ | 约 13.2 万对「team1 vs team2」的获胜概率（team1 的 Pred） |
| 提交格式？ | `ID = 2026_XXXX_YYYY`，`Pred = P(XXXX 胜)`，XXXX < YYYY |
| NCAA 赛制？ | 68 队 → Play-in → 64 队单败淘汰 6 轮 |
| Substack 数据？ | 每周更新的 projection table（XLSX），含 rd1_win～rd6_win |
| 建议下载？ | 男子/女子最新 projection 各一；提交前优先用 Selection Sunday 之后那一期 |
| 如何使用？ | 解析 XLSX → 映射 TeamID → 结合 bracket 推导 pairwise 胜率 → 填入 submission |

---

## 参考

- 竞赛数据说明：`data_explained`
- 深度分析报告：`docs/NCAA2026_深度分析报告.md`
- goto_conversion：[GitHub](https://github.com/gotoConversion/goto_conversion)、[Substack](https://gotoconversion.substack.com/archive)
- 项目工具：`src/goto_utils.py`、`src/optimal_strategy.py`
