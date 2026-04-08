# 从 goto_conversion 概率表生成 Kaggle Submission

## 一、你的概率表结构（Womensmarchmadness_20260307_081616.xlsx）

### 1.1 基本信息

| 属性 | 值 |
|------|-----|
| 行数 | 64 |
| 列 | Team, Elo_Rating, rd1_win, rd2_win, rd3_win, rd4_win, rd5_win, rd6_win |
| 队伍范围 | **预测进入锦标赛正赛的 64 支女篮队伍** |

### 1.2 这 64 支队伍代表什么？

**不是**「未被淘汰的队」，而是 **Selection Sunday 之前** 对「谁会进锦标赛、以及 bracket 如何排布」的预测。

- 日期 20260307：2026 年 3 月 7 日（约 Selection Sunday 前一周）
- 此时锦标赛未开打，表中是 **预测能进正赛的 64 队** 及其各轮晋级概率
- 实际 NCAA 女子为 68 队（含 4 场 play-in），本表可理解为已确定/预测出的「主 bracket 64 队」

### 1.3 表中 64 队列表（示例）

```
UConn Huskies, Colorado Buffaloes, Arizona State Sun Devils, California Golden Bears,
Baylor Bears, Arizona Wildcats, Maryland Terrapins, Gonzaga Bulldogs, Duke Blue Devils,
Arkansas Razorbacks, North Carolina Tar Heels, Syracuse Orange, Iowa Hawkeyes,
Georgia Tech Yellow Jackets, Ohio State Buckeyes, Illinois Fighting Illini,
Texas Longhorns, Clemson Tigers, Michigan State Spartans, Florida State Seminoles,
Iowa State Cyclones, South Dakota State Jackrabbits, West Virginia Mountaineers,
Florida Gators, LSU Tigers, Georgia Lady Bulldogs, Washington Huskies, Stanford Cardinal,
TCU Horned Frogs, Miami (FL) Hurricanes, North Carolina State Wolfpack,
Grand Canyon Antelopes, UCLA Bruins, Creighton Bluejays, Richmond Spiders,
Virginia Tech Hokies, Kentucky Wildcats, Columbia Lions, Ole Miss Rebels,
Texas A&M Aggies, Michigan Wolverines, Mississippi State Bulldogs, Alabama Crimson Tide,
Oregon Ducks, Oklahoma Sooners, Indiana Hoosiers, Notre Dame Fighting Irish,
Nebraska Cornhuskers, South Carolina Gamecocks, Utah Utes, Villanova Wildcats,
Kansas State Wildcats, Texas Tech Lady Raiders, Wisconsin Badgers,
Tennessee Lady Volunteers, Marquette Golden Eagles, Vanderbilt Commodores,
Missouri Tigers, Minnesota Golden Gophers, Oklahoma State Cowgirls,
Louisville Cardinals, Rutgers Scarlet Knights, USC Trojans, Princeton Tigers
```

### 1.4 表内 bracket 结构

**相邻两行 = 首轮对阵**，rd1_win 直接为单场胜率：

| 行 | 队 A | 队 B | rd1_win 含义 |
|----|------|------|--------------|
| 0-1 | UConn | Colorado | P(UConn 胜) ≈ 0.984, P(Colorado 胜) ≈ 0.016 |
| 2-3 | Arizona State | California | 各约 0.5 |
| 4-5 | Baylor | Arizona | P(Baylor 胜) ≈ 0.68 |
| … | … | … | … |

---

## 二、完整转换流程

### 2.1 整体逻辑

```
概率表 (Team, rd1_win~rd6_win)
    ↓ 队名 → TeamID（WTeamSpellings）
    ↓ 对每对 (team1, team2)：判断相遇轮次 r（get_roundOfMatch）
    ↓ 用 rd_r_win 计算 P(team1 胜 team2)
SampleSubmissionStage2 格式的 submission
```

### 2.2 核心公式

当 team1 与 team2 可能在 **第 r 轮** 相遇时：

```
P(team1 胜 team2) ≈ rd_r_win(team1) / [ rd_r_win(team1) + rd_r_win(team2) ]
```

- r=1：首轮对阵，也可直接用表中该对的 rd1_win
- r≥2：用对应轮的 rd_r_win 做上述 Bradley-Terry 式近似

### 2.3 需要的输入

| 文件 | 来源 | 用途 |
|------|------|------|
| 概率表 XLSX/CSV | Substack 下载（Selection Sunday 后最新） | rd*_win、Elo_Rating |
| SampleSubmissionStage2.csv | Kaggle 竞赛数据 | ID 列表、格式 |
| WTeamSpellings.csv | march-machine-learning-mania-2026 | 队名 → TeamID |
| WNCAATourneySeeds.csv | 同上（含 2026） | 种子、bracket 结构，算相遇轮次 |

### 2.4 边界情况

1. **队名不一致**：Substack 用 "UConn Huskies"，Kaggle 用 "uconn"、"connecticut"，需建映射表
2. **不在概率表中的队**：若某队不在 64 队中，可设 Pred=0.5 或按种子/强弱给默认值
3. **不同区、只在 Final Four 才可能相遇**：用 rd5_win 或 rd6_win 的近似

---

## 三、实施步骤（Selection Sunday 后）

### Step 1：获取最新概率表

- 在 Selection Sunday（2026-03-15）之后，从 Substack 下载最新女篮 projection table
- 格式与当前 XLSX 一致：Team, Elo_Rating, rd1_win~rd6_win

### Step 2：建立 Team → TeamID 映射

用 WTeamSpellings 做模糊匹配或手工映射，例如：

```
UConn Huskies → 3163 (connecticut)
Colorado Buffaloes → 3160 (colorado)
...
```

### Step 3：解析 bracket 与首轮对阵

- 表中行序对应 bracket，相邻行为首轮对阵
- 得到首轮 32 对的 (teamA, teamB, P_A_win)

### Step 4：对 SampleSubmissionStage2 逐行填 Pred

对每个 ID = `2026_XXXX_YYYY`：

1. 解析 team1_id=XXXX, team2_id=YYYY
2. 用 seeds 计算 `r = get_roundOfMatch(team1, team2, seeds_df)`
3. 查概率表：
   - 若 r=1 且 (team1, team2) 为表中首轮对：用对应 rd1_win
   - 否则：`pred = rd_r_win(team1) / (rd_r_win(team1) + rd_r_win(team2))`
4. 若任一方不在表中：Pred=0.5
5. 确保 team1 为 ID 中较小 TeamID 的队，Pred 表示该队胜率

### Step 5：后处理（可选）

- 使用 `set_optimalStrategy` 调整部分对阵
- 做 `apply_manual_overrides` 等

---

## 四、注意事项

1. **选表时间**：尽量用 Selection Sunday 之后、且标注使用「完整算法」的 projection
2. **男子 vs 女子**：需分别下载男女概率表，分别处理
3. **Play-in**：若表只有 64 队，play-in 的 4 个 slot 可能需额外逻辑或默认 0.5
4. **队名映射**：Substack 与 Kaggle 命名差异较大，映射表需要仔细校对

---

## 五、参考代码位置

- 轮次推断：`src/optimal_strategy.py` → `get_roundOfMatch`
- 队名映射：`march-machine-learning-mania-2026/WTeamSpellings.csv`
- 策略覆盖：`src/optimal_strategy.py` → `set_optimalStrategy`, `apply_manual_overrides`
