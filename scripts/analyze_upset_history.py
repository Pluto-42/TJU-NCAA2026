#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 Duke、Michigan、Arizona 三队历史锦标赛表现
重点：是否曾被爆冷淘汰（作为高种子输给低种子）
"""

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "march-machine-learning-mania-2026")

# 三队 TeamID
TEAMS = {"Duke": 1181, "Michigan": 1276, "Arizona": 1112}


def parse_seed(seed_str: str) -> int:
    """从 Seed 如 'W01' 提取数字部分"""
    if pd.isna(seed_str):
        return 99
    s = str(seed_str).strip()
    if len(s) >= 3 and s[-2:].isdigit():
        return int(s[-2:])
    if s[-1].isdigit():
        return int(s[-1])
    return 99


def main():
    results = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneyCompactResults.csv"))
    seeds = pd.read_csv(os.path.join(DATA_DIR, "MNCAATourneySeeds.csv"))

    # DayNum -> 轮次
    def round_name(day):
        if day in (134, 135):
            return "Play-in"
        if day == 136:
            return "R1"  # 首轮
        if day == 137:
            return "R2"  # 次轮
        if day == 138:
            return "R32"
        if day == 139:
            return "S16"
        if day in (143, 144):
            return "E8"
        if day in (145, 146):
            return "F4"
        if day in (152, 154):
            return "Final"
        return f"D{day}"

    seeds["SeedNum"] = seeds["Seed"].apply(parse_seed)
    seed_lookup = seeds.set_index(["Season", "TeamID"])["SeedNum"].to_dict()

    # 收集每队的失利记录
    losses = []
    for season in results["Season"].unique():
        sel = results[results["Season"] == season]
        for _, r in sel.iterrows():
            w, l = r["WTeamID"], r["LTeamID"]
            day = r["DayNum"]
            for name, tid in TEAMS.items():
                if tid == l:
                    w_seed = seed_lookup.get((season, w), 99)
                    l_seed = seed_lookup.get((season, l), 99)
                    upset = l_seed < w_seed  # 高种子输给低种子
                    losses.append({
                        "Team": name,
                        "Season": season,
                        "Round": round_name(day),
                        "DayNum": day,
                        "LostTo": w,
                        "LoserSeed": l_seed,
                        "WinnerSeed": w_seed,
                        "Upset": upset,
                    })

    df = pd.DataFrame(losses)
    if df.empty:
        print("无失利记录")
        return

    # 加载队名
    mteams = pd.read_csv(os.path.join(DATA_DIR, "MTeams.csv"))
    name_map = dict(zip(mteams["TeamID"], mteams["TeamName"]))
    df["LostToName"] = df["LostTo"].map(name_map)

    # 只保留最近 6 年（2020–2025）
    df_recent = df[df["Season"] >= 2020].copy()

    # 输出
    print("=" * 70)
    print("Duke / Michigan / Arizona 历史锦标赛失利记录（最近 6 年：2020–2025）")
    print("=" * 70)

    for name in ["Duke", "Michigan", "Arizona"]:
        sub = df_recent[df_recent["Team"] == name].sort_values("Season", ascending=False)
        if sub.empty:
            print(f"\n【{name}】无 2000 年后淘汰记录（或数据中未参赛）")
            continue

        upsets = sub[sub["Upset"]]
        print(f"\n【{name}】共 {len(sub)} 次淘汰，其中 {len(upsets)} 次为爆冷（高种子输低种子）")
        print("-" * 60)
        for _, r in sub.iterrows():
            u = "⚠️ 爆冷" if r["Upset"] else ""
            print(f"  {r['Season']} {r['Round']}: 输给 #{r['WinnerSeed']} {r['LostToName']} "
                  f"(自身 #{r['LoserSeed']}) {u}")
        if not upsets.empty:
            print(f"  → 爆冷多发生在前几轮，谨慎对弱队给过高预测")

    # 汇总：谁最不稳定
    print("\n" + "=" * 70)
    print("爆冷统计（高种子输给低种子）")
    print("=" * 70)
    for name in ["Duke", "Michigan", "Arizona"]:
        sub = df_recent[df_recent["Team"] == name]
        up = sub[sub["Upset"]]
        total = len(sub)
        pct = 100 * len(up) / total if total > 0 else 0
        print(f"  {name}: {len(up)}/{total} 次淘汰为爆冷 ({pct:.0f}%)")

    # 保存
    out = os.path.join(ROOT, "docs", "三队历史爆冷分析.csv")
    df_recent.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out}")


if __name__ == "__main__":
    main()
