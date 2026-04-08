#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Texas Tech 男篮完整对阵预测表

从 matchupProbabilities_men.csv 提取 Texas Tech (TeamID 1403) 对所有可能的对手的胜率预测，
并标注对手的 pairwise 排名，便于对强队对阵时做下调决策。
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TT_TEAM_ID = 1403


def main():
    # 加载对阵概率
    mp_path = os.path.join(ROOT, "matchupProbabilities_men.csv")
    mp = pd.read_csv(mp_path)

    # 筛选 Texas Tech 参与的对阵
    tt_as_t1 = mp[mp["T1_TeamID"] == TT_TEAM_ID].copy()
    tt_as_t2 = mp[mp["T2_TeamID"] == TT_TEAM_ID].copy()

    rows = []
    for _, r in tt_as_t1.iterrows():
        rows.append({
            "OppID": r["T2_TeamID"],
            "OppName": r["T2_TeamName"],
            "TT_WinProb": r["Pred"],
        })
    for _, r in tt_as_t2.iterrows():
        rows.append({
            "OppID": r["T1_TeamID"],
            "OppName": r["T1_TeamName"],
            "TT_WinProb": 1.0 - r["Pred"],
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["OppID"])

    # 加载 pairwise 排名
    rank_path = os.path.join(ROOT, "docs", "team_ranking_men_pairwise.csv")
    ranks = pd.read_csv(rank_path)
    rank_map = dict(zip(ranks["TeamID"], ranks["PairwiseRank"]))

    df["OppRank"] = df["OppID"].map(rank_map)
    df["OppRankStr"] = df["OppRank"].apply(lambda x: f"#{x}" if pd.notna(x) else "—")

    # 按对手排名排序（有排名的按排名升序，未上榜放最后）
    df["_sort"] = df["OppRank"].fillna(999)
    df = df.sort_values("_sort").drop(columns=["_sort"])

    # 强弱分类：前10强队 / 11-25 中强 / 26+ 中弱或弱队
    def strength_group(rank):
        if pd.isna(rank):
            return "未上榜"
        r = int(rank)
        if r <= 10:
            return "强队(前10)"
        if r <= 25:
            return "中强(11-25)"
        return "中弱/弱队(26+)"

    df["对手档位"] = df["OppRank"].apply(strength_group)

    # 伤病下调：强队 -7%，中强 -4%，中弱/弱 -2.5%
    def adj_drop(group):
        if group == "强队(前10)":
            return 0.07
        if group == "中强(11-25)":
            return 0.04
        return 0.025  # 中弱/弱队 2-3% 取 2.5%
    df["下调幅度"] = df["对手档位"].apply(adj_drop)
    df["TT_WinProb_原"] = df["TT_WinProb"]
    df["TT_WinProb_下调后"] = (df["TT_WinProb"] - df["下调幅度"]).clip(0, 1)

    # 输出原始表 CSV
    out_csv = os.path.join(ROOT, "docs", "Texas_Tech_对阵预测完整表.csv")
    df[["OppName", "OppRankStr", "对手档位", "TT_WinProb"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )
    print(f"原始表 CSV 已保存: {out_csv}\n")

    # 输出伤病下调版 CSV 和 Markdown
    out_adj_csv = os.path.join(ROOT, "docs", "Texas_Tech_对阵预测_伤病下调版.csv")
    df_export = df[["OppName", "OppRankStr", "对手档位", "下调幅度", "TT_WinProb_原", "TT_WinProb_下调后"]].copy()
    df_export.to_csv(out_adj_csv, index=False, encoding="utf-8-sig")
    print(f"伤病下调版 CSV 已保存: {out_adj_csv}\n")

    # 完整打印（伤病下调版）
    print("=" * 100)
    print("Texas Tech 男篮 - 伤病下调版对阵预测（pairwise #19，强队-7% / 中强-4% / 中弱弱-2.5%）")
    print("=" * 100)
    print(f"{'对手':<20} {'对手排名':>6} {'档位':>12} {'下调':>6} {'原胜率':>8} {'下调后胜率':>10}")
    print("-" * 100)

    for _, r in df.iterrows():
        print(f"{r['OppName']:<20} {r['OppRankStr']:>6} {r['对手档位']:>12} {r['下调幅度']:>6.1%} {r['TT_WinProb_原']:>8.2%} {r['TT_WinProb_下调后']:>10.2%}")

    print("-" * 100)
    print(f"共 {len(df)} 个潜在对手")

    # 输出 Markdown（伤病下调版）
    md_path = os.path.join(ROOT, "docs", "Texas_Tech_对阵预测_伤病下调版.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Texas Tech 男篮 - 伤病下调版对阵预测\n\n")
        f.write("pairwise 排名 **#19**，JT Toppin ACL 伤缺。下调规则：强队(前10) -7%，中强(11-25) -4%，中弱/弱队(26+) -2.5%\n\n")
        f.write("| 对手 | 对手排名 | 档位 | 下调幅度 | 原胜率 | 下调后胜率 |\n")
        f.write("|------|----------|------|----------|--------|------------|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['OppName']} | {r['OppRankStr']} | {r['对手档位']} | {r['下调幅度']:.1%} | {r['TT_WinProb_原']:.2%} | {r['TT_WinProb_下调后']:.2%} |\n")
    print(f"\n伤病下调版 Markdown 已保存: {md_path}")


if __name__ == "__main__":
    main()
