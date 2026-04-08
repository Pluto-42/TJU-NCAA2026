#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
伤病影响分析脚本

功能：
1. 将伤病日期映射到 Kaggle MSeasons 的 DayNum
2. 按受伤日期切分每队伤病前/后的比赛
3. 统计伤病前后胜率、场均净胜分
4. 列出伤后输给的对手
5. 用伤后样本预估「伤病版」实力（简化版：胜率、场均净胜分）

伤病信息（已确认日期）：
- Texas Tech 男 (1403): JT Toppin ACL 撕裂，2026-02-17（对 Arizona State）
- Duke 男 (1181): Caleb Foster 脚部骨折，2026-03-07（对 UNC 上半场）；Pat Ngongba 脚伤，3/7 对 UNC 未上
- USC 女 (3425): Jazzy Davidson 右肩伤，2026-03-05（Big Ten 锦标赛对 Washington）
"""

import os
import pandas as pd
from datetime import datetime

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "march-machine-learning-mania-2026")

# 2026 赛季 DayZero = 11/03/2025
DAY_ZERO_2026 = datetime(2025, 11, 3)


def date_to_daynum(d: datetime) -> int:
    """将日期转换为 2026 赛季 DayNum（自 DayZero 起的天数）"""
    return (d - DAY_ZERO_2026).days


# 伤病配置：(TeamID, 伤病日, 描述, 性别 M/W)
INJURY_CONFIG = [
    # Texas Tech: Toppin 2/17 受伤
    (1403, date_to_daynum(datetime(2026, 2, 17)), "JT Toppin ACL 撕裂", "M"),
    # Duke: Foster 3/7 受伤（该场仍打完上半场，下一场起缺阵）；Ngongba 同日未上
    (1181, date_to_daynum(datetime(2026, 3, 7)), "Caleb Foster + Pat Ngongba 脚伤", "M"),
    # USC 女: Davidson 3/5 受伤
    (3425, date_to_daynum(datetime(2026, 3, 5)), "Jazzy Davidson 右肩伤", "W"),
]


def load_games(season: int, gender: str) -> pd.DataFrame:
    """加载指定赛季的常规赛+联盟锦标赛数据（Kaggle 数据通常包含联盟锦标赛）"""
    fname = "MRegularSeasonDetailedResults.csv" if gender == "M" else "WRegularSeasonDetailedResults.csv"
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path)
    df = df[df["Season"] == season].copy()
    return df


def get_team_games(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """
    提取某队参与的所有比赛，并标准化为：TeamID, OppID, Score, OppScore, DayNum, IsWin
    """
    rows = []
    for _, r in df.iterrows():
        w, l = r["WTeamID"], r["LTeamID"]
        wscore, lscore = r["WScore"], r["LScore"]
        day = r["DayNum"]
        if w == team_id:
            rows.append({"TeamID": team_id, "OppID": l, "Score": wscore, "OppScore": lscore, "DayNum": day, "IsWin": 1})
        elif l == team_id:
            rows.append({"TeamID": team_id, "OppID": w, "Score": lscore, "OppScore": wscore, "DayNum": day, "IsWin": 0})
    return pd.DataFrame(rows)


def analyze_team(team_id: int, injury_day: int, label: str, gender: str) -> dict:
    """
    分析单支队伍的伤病前后表现。
    返回：before/after 的胜场、总场次、胜率、场均净胜分；伤后输球对手列表；伤病前后每场明细。
    """
    df = load_games(2026, gender)
    games = get_team_games(df, team_id)
    if games.empty:
        return {"team_id": team_id, "label": label, "error": "无 2026 数据"}

    before = games[games["DayNum"] < injury_day].sort_values("DayNum")
    after = games[games["DayNum"] >= injury_day].sort_values("DayNum")

    def stats(g: pd.DataFrame) -> dict:
        if g.empty:
            return {"wins": 0, "games": 0, "win_rate": 0.0, "avg_margin": 0.0}
        wins = g["IsWin"].sum()
        n = len(g)
        margin = (g["Score"] - g["OppScore"]).mean()
        return {"wins": int(wins), "games": n, "win_rate": round(wins / n, 4), "avg_margin": round(margin, 2)}

    before_s = stats(before)
    after_s = stats(after)

    # 伤后输球对手
    losses_after = after[after["IsWin"] == 0]
    loss_opponents = (
        losses_after["OppID"].tolist()
        if not losses_after.empty
        else []
    )

    # 伤病前后每场明细：DayNum, OppID, Result, Score, OppScore
    def game_rows(g: pd.DataFrame) -> list:
        if g.empty:
            return []
        return [
            {"DayNum": int(r["DayNum"]), "OppID": int(r["OppID"]), "Result": "胜" if r["IsWin"] else "负", "Score": int(r["Score"]), "OppScore": int(r["OppScore"])}
            for _, r in g.iterrows()
        ]

    return {
        "team_id": team_id,
        "label": label,
        "injury_day": injury_day,
        "before": before_s,
        "after": after_s,
        "before_games": game_rows(before),
        "after_games": game_rows(after),
        "loss_opponents": loss_opponents,
        "loss_opp_names": [],  # 稍后填队名
    }


def load_team_names() -> tuple[dict, dict]:
    """加载 MTeams 和 WTeams 的 TeamID -> 队名映射"""
    m, w = {}, {}
    for fname, d in [("MTeams.csv", m), ("WTeams.csv", w)]:
        path = os.path.join(DATA_DIR, fname)
        t = pd.read_csv(path)
        for _, r in t.iterrows():
            d[r["TeamID"]] = r["TeamName"]
    return m, w


def load_pairwise_ranks() -> tuple[dict, dict]:
    """加载男篮/女篮 pairwise 排名：TeamID -> PairwiseRank（未上榜则返回 None）"""
    m_rank, w_rank = {}, {}
    for fname, d in [("team_ranking_men_pairwise.csv", m_rank), ("team_ranking_women_pairwise.csv", w_rank)]:
        path = os.path.join(ROOT, "docs", fname)
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            d[r["TeamID"]] = int(r["PairwiseRank"])
    return m_rank, w_rank


def main():
    m_names, w_names = load_team_names()

    results = []
    for team_id, injury_day, label, gender in INJURY_CONFIG:
        r = analyze_team(team_id, injury_day, label, gender)
        if "error" in r:
            results.append(r)
            continue

        # 填伤后输球对手队名，以及每场对手队名
        name_map = m_names if gender == "M" else w_names
        r["loss_opp_names"] = [name_map.get(o, f"Team_{o}") for o in r["loss_opponents"]]
        r["team_name"] = name_map.get(team_id, f"Team_{team_id}")
        r["gender"] = gender
        for g in r.get("before_games", []) + r.get("after_games", []):
            g["OppName"] = name_map.get(g["OppID"], f"Team_{g['OppID']}")
        results.append(r)

    # 加载 pairwise 排名（用于展示对阵双方实力档位）
    m_rank, w_rank = load_pairwise_ranks()

    # 生成赛程对阵表（伤病前/后 每场对手、结果、比分、 pairwise 排名）
    table_rows = []
    for r in results:
        if "error" in r or "before_games" not in r:
            continue
        tn = r["team_name"]
        tid = r["team_id"]
        rank_map = m_rank if r["gender"] == "M" else w_rank
        team_rank = rank_map.get(tid)
        team_rank_str = str(team_rank) if team_rank is not None else "—"

        for g in r["before_games"]:
            opp_rank = rank_map.get(g["OppID"])
            opp_rank_str = str(opp_rank) if opp_rank is not None else "—"
            table_rows.append({
                "队伍": tn, "队伍排名": team_rank_str, "时段": "伤病前", "DayNum": g["DayNum"],
                "对手": g["OppName"], "对手排名": opp_rank_str,
                "结果": g["Result"], "本方得分": g["Score"], "对手得分": g["OppScore"],
            })
        for g in r["after_games"]:
            opp_rank = rank_map.get(g["OppID"])
            opp_rank_str = str(opp_rank) if opp_rank is not None else "—"
            table_rows.append({
                "队伍": tn, "队伍排名": team_rank_str, "时段": "伤病后", "DayNum": g["DayNum"],
                "对手": g["OppName"], "对手排名": opp_rank_str,
                "结果": g["Result"], "本方得分": g["Score"], "对手得分": g["OppScore"],
            })

    if table_rows:
        tbl_df = pd.DataFrame(table_rows)
        tbl_path = os.path.join(ROOT, "docs", "伤病前后赛程明细表.csv")
        tbl_df.to_csv(tbl_path, index=False, encoding="utf-8-sig")
        print(f"赛程明细表已保存至: {tbl_path}")

    # 控制台输出
    print("=" * 80)
    print("伤病影响分析报告 (2026 赛季)")
    print("=" * 80)

    for r in results:
        if "error" in r:
            print(f"\n{r['label']} ({r['team_id']}): {r['error']}")
            continue

        print(f"\n【{r['team_name']}】 {r['label']}")
        print(f"  伤病日对应 DayNum: {r['injury_day']}")
        b, a = r["before"], r["after"]
        print(f"  伤病前: {b['wins']}胜{b['games']-b['wins']}负 共{b['games']}场, 胜率 {b['win_rate']:.2%}, 场均净胜 {b['avg_margin']:+.1f}")
        print(f"  伤病后: {a['wins']}胜{a['games']-a['wins']}负 共{a['games']}场, 胜率 {a['win_rate']:.2%}, 场均净胜 {a['avg_margin']:+.1f}")

        if r["loss_opponents"]:
            print(f"  伤后输给: {', '.join(r['loss_opp_names'])}")
        else:
            print("  伤后无输球")

        # 伤后实力预估（简化为胜率 + 场均净胜的定性）
        if a["games"] >= 1:
            est = f"胜率{a['win_rate']:.1%} / 场均净胜{a['avg_margin']:.1f}"
            print(f"  伤后实力预估: {est}")

    # 保存为 CSV（摘要）
    out_path = os.path.join(ROOT, "docs", "injury_impact_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for r in results:
        if "error" in r:
            continue
        rows.append({
            "Team": r["team_name"],
            "Injury": r["label"],
            "Before_W": r["before"]["wins"],
            "Before_G": r["before"]["games"],
            "Before_WinRate": r["before"]["win_rate"],
            "Before_Margin": r["before"]["avg_margin"],
            "After_W": r["after"]["wins"],
            "After_G": r["after"]["games"],
            "After_WinRate": r["after"]["win_rate"],
            "After_Margin": r["after"]["avg_margin"],
            "Loss_To": "; ".join(r["loss_opp_names"]),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n摘要已保存至: {out_path}")

    # 输出 Markdown 报告（含赛程对阵表）
    md_path = os.path.join(ROOT, "docs", "伤病影响分析报告.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 伤病影响分析报告（2026 NCAA）\n\n")
        f.write("## 伤病日期与 DayNum 对应\n\n")
        f.write("| 队伍 | 伤病 | 发生日期 | DayNum |\n")
        f.write("|------|------|----------|--------|\n")
        f.write("| Texas Tech 男 | JT Toppin ACL 撕裂 | 2026-02-17 | 106 |\n")
        f.write("| Duke 男 | Caleb Foster 脚部骨折 + Pat Ngongba 脚伤 | 2026-03-07 | 124 |\n")
        f.write("| USC 女 | Jazzy Davidson 右肩伤 | 2026-03-05 | 122 |\n\n")

        f.write("## 伤病前后赛程对阵表\n\n")
        f.write("排名为 pairwise 预测排名（未上榜显示 —）\n\n")
        for r in results:
            if "error" in r or "before_games" not in r:
                continue
            rank_map = m_rank if r["gender"] == "M" else w_rank
            tr = rank_map.get(r["team_id"])
            tr_str = str(tr) if tr is not None else "—"
            f.write(f"### {r['team_name']}（排名 #{tr_str}） - {r['label']}\n\n")
            f.write("| 时段 | DayNum | 对手 | 对手排名 | 结果 | 本方得分 | 对手得分 |\n")
            f.write("|------|--------|------|----------|------|----------|----------|\n")
            for g in r.get("before_games", []):
                orank = rank_map.get(g["OppID"])
                orank_str = f"#{orank}" if orank is not None else "—"
                f.write(f"| 伤病前 | {g['DayNum']} | {g['OppName']} | {orank_str} | {g['Result']} | {g['Score']} | {g['OppScore']} |\n")
            for g in r.get("after_games", []):
                orank = rank_map.get(g["OppID"])
                orank_str = f"#{orank}" if orank is not None else "—"
                f.write(f"| 伤病后 | {g['DayNum']} | {g['OppName']} | {orank_str} | {g['Result']} | {g['Score']} | {g['OppScore']} |\n")
            f.write("\n")

        f.write("## 各队伤病前后表现（汇总）\n\n")
        for r in results:
            if "error" in r:
                continue
            b, a = r["before"], r["after"]
            f.write(f"### {r['team_name']} - {r['label']}\n\n")
            f.write(f"- **伤病前**: {b['wins']}胜{int(b['games']-b['wins'])}负，胜率 {b['win_rate']:.2%}，场均净胜 {b['avg_margin']:+.1f}\n")
            f.write(f"- **伤病后**: {a['wins']}胜{int(a['games']-a['wins'])}负，胜率 {a['win_rate']:.2%}，场均净胜 {a['avg_margin']:+.1f}\n")
            if r["loss_opp_names"]:
                f.write(f"- **伤后输给**: {', '.join(r['loss_opp_names'])}\n")
            f.write("\n")

    print(f"Markdown 报告已保存至: {md_path}")


if __name__ == "__main__":
    main()
