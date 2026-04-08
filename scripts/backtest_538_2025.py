#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025 538 submission 在 2025 真实锦标赛数据上的 Brier 回测

用法：python scripts/backtest_538_2025.py

输入：
  - 2025_538/submission.csv：538 的预测（ID, Pred）
  - march-machine-learning-mania-2026/MNCAATourneyCompactResults.csv：男篮真实赛果
  - march-machine-learning-mania-2026/WNCAATourneyCompactResults.csv：女篮真实赛果

Brier 定义：对每场实际发生的比赛，(Pred - actual)^2，其中 actual=1 表示 ID 中较小 TeamID 的队获胜。
Kaggle 惯例：ID = Season_T1_T2，T1<T2，Pred = P(T1 胜 T2)。
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "march-machine-learning-mania-2026")
SUB_PATH = os.path.join(ROOT, "2025_538", "submission.csv")
M_TOURNEY = os.path.join(DATA_DIR, "MNCAATourneyCompactResults.csv")
W_TOURNEY = os.path.join(DATA_DIR, "WNCAATourneyCompactResults.csv")
SEASON = 2025


def get_submission_id(w_team: int, l_team: int) -> str:
    """
    由胜队 W、负队 L 生成 Kaggle submission ID。
    ID = 2025_T1_T2，T1 < T2，Pred = P(T1 胜 T2)。
    当 T1 获胜时 actual=1，否则 actual=0。
    """
    t1, t2 = min(w_team, l_team), max(w_team, l_team)
    return f"{SEASON}_{t1}_{t2}"


def get_actual(w_team: int, l_team: int) -> int:
    """actual=1 表示较小 TeamID 获胜，否则 0"""
    return 1 if w_team < l_team else 0


def main():
    # 1. 加载 submission
    sub = pd.read_csv(SUB_PATH)
    sub_dict = dict(zip(sub["ID"].astype(str), sub["Pred"]))

    # 2. 加载 2025 锦标赛真实赛果
    m_res = pd.read_csv(M_TOURNEY)
    w_res = pd.read_csv(W_TOURNEY)
    m_2025 = m_res[m_res["Season"] == SEASON]
    w_2025 = w_res[w_res["Season"] == SEASON]

    rows = []
    for _, r in m_2025.iterrows():
        wid, lid = r["WTeamID"], r["LTeamID"]
        sid = get_submission_id(wid, lid)
        actual = get_actual(wid, lid)
        pred = sub_dict.get(sid)
        if pred is None:
            pred = 0.5  # 缺失时用 0.5
        rows.append({"gender": "M", "ID": sid, "WTeamID": wid, "LTeamID": lid, "actual": actual, "pred": pred})

    for _, r in w_2025.iterrows():
        wid, lid = r["WTeamID"], r["LTeamID"]
        sid = get_submission_id(wid, lid)
        actual = get_actual(wid, lid)
        pred = sub_dict.get(sid)
        if pred is None:
            pred = 0.5
        rows.append({"gender": "W", "ID": sid, "WTeamID": wid, "LTeamID": lid, "actual": actual, "pred": pred})

    df = pd.DataFrame(rows)

    # 3. 计算 Brier
    df["brier"] = (df["pred"] - df["actual"]) ** 2
    total_brier = df["brier"].mean()
    n = len(df)

    m_df = df[df["gender"] == "M"]
    w_df = df[df["gender"] == "W"]
    m_brier = m_df["brier"].mean() if len(m_df) > 0 else 0.0
    w_brier = w_df["brier"].mean() if len(w_df) > 0 else 0.0

    # 4. 输出
    print("=" * 70)
    print("2025 538 Submission 回测结果（2025 真实锦标赛）")
    print("=" * 70)
    print(f"Submission 文件: {SUB_PATH}")
    print(f"总场次: {n} 场（男篮 {len(m_df)} 场，女篮 {len(w_df)} 场）")
    print()
    print(f"男篮 Brier: {m_brier:.6f}")
    print(f"女篮 Brier: {w_brier:.6f}")
    print(f"总体 Brier: {total_brier:.6f}")
    print("=" * 70)

    # 保存明细
    out_path = os.path.join(ROOT, "docs", "backtest_538_2025_results.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out_path}")


if __name__ == "__main__":
    main()
