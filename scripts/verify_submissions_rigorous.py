#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三套提交严谨验证脚本

1. 校验数据引用、TeamID、Seeds 是否正确
2. 抽样检查关键逻辑
3. 以 Duke 为例，打印其赛事信息（尤其 E8/F4/Final 最后几轮）
4. 核对 submission 与基准的一致性
"""

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.optimal_strategy import get_tourneyFlag, get_roundOfMatch

# 常量（与 generate_three_submissions 一致）
DUKE_ID = 1181
MICHIGAN_ID = 1276
ARIZONA_ID = 1112
TT_ID = 1403
ROUND_NAMES = {
    1: "Play-in",
    2: "R64",
    3: "R32",
    4: "S16",
    5: "E8",
    6: "F4",
    7: "Final",
}


def verify_seeds_and_ids(mens_seeds):
    """校验 2026 seeds 中 Duke/Michigan/Arizona/Texas Tech 的 TeamID 与种子"""
    print("=" * 70)
    print("【1】Seeds 与 TeamID 校验")
    print("=" * 70)
    expected = {
        DUKE_ID: ("Duke", "W01"),
        MICHIGAN_ID: ("Michigan", "Y01"),
        ARIZONA_ID: ("Arizona", "Z01"),
        TT_ID: ("Texas Tech", "Y05"),
    }
    for tid, (name, exp_seed) in expected.items():
        row = mens_seeds[mens_seeds["TeamID"] == tid]
        if len(row) == 0:
            print(f"  [ERR] {name} (ID={tid}) 未出现在 2026 seeds 中!")
        else:
            seed = row["Seed"].values[0]
            ok = "OK" if seed == exp_seed else "CHECK"
            print(f"  {name:12} TeamID={tid}  Seed={seed} (预期 {exp_seed}) [{ok}]")
    print()


def verify_file_paths():
    """校验所有关键文件路径存在"""
    print("=" * 70)
    print("【2】关键文件路径校验")
    print("=" * 70)
    paths = [
        ("matchupProbabilities_men.csv", os.path.join(ROOT, "matchupProbabilities_men.csv")),
        ("matchupProbabilities.csv", os.path.join(ROOT, "matchupProbabilities.csv")),
        ("SampleSubmissionStage2.csv", os.path.join(ROOT, "march-machine-learning-mania-2026", "SampleSubmissionStage2.csv")),
        ("MNCAATourneySeeds.csv", os.path.join(ROOT, "march-machine-learning-mania-2026", "MNCAATourneySeeds.csv")),
        ("Texas_Tech_对阵预测_伤病下调版.csv", os.path.join(ROOT, "docs", "Texas_Tech_对阵预测_伤病下调版.csv")),
        ("submission_A_Duke.csv", os.path.join(ROOT, "outputs", "submission_A_Duke.csv")),
    ]
    all_ok = True
    for name, p in paths:
        exists = os.path.exists(p)
        all_ok = all_ok and exists
        print(f"  {name:45} {'[OK]' if exists else '[MISSING]'}")
    if not all_ok:
        print("  [ERR] 存在缺失文件!")
    print()
    return all_ok


def build_duke_matchups_by_round(mens_seeds, id2name, base_pred_map, sub_pred_map):
    """收集 Duke 参与的所有对阵，按轮次分组，含 base/sub Pred"""
    duke_games = []
    seed_ids = set(mens_seeds["TeamID"].tolist())
    for uid, pred_sub in sub_pred_map.items():
        if str(uid).startswith("2026_"):
            parts = str(uid).split("_")
            if len(parts) != 3:
                continue
            t1, t2 = int(parts[1]), int(parts[2])
            if t1 + t2 > 6000:
                continue
            if DUKE_ID not in (t1, t2):
                continue
            if t1 not in seed_ids or t2 not in seed_ids:
                continue
            rnd = get_tourneyFlag(t1, t2, mens_seeds)
            if rnd == 0:
                continue
            pred_base = base_pred_map.get(uid, None)
            opp_id = t2 if t1 == DUKE_ID else t1
            opp_name = id2name.get(opp_id, f"Team_{opp_id}")
            duke_win_sub = pred_sub if t1 == DUKE_ID else (1.0 - pred_sub)
            duke_win_base = None
            if pred_base is not None:
                duke_win_base = pred_base if t1 == DUKE_ID else (1.0 - pred_base)
            duke_games.append({
                "round": rnd,
                "ID": uid,
                "opp_id": opp_id,
                "opp_name": opp_name,
                "duke_win_base": duke_win_base,
                "duke_win_sub": duke_win_sub,
                "pred_base": pred_base,
                "pred_sub": pred_sub,
            })
    return duke_games


def print_duke_tournament_path(duke_games, round_order=(5, 6, 7)):
    """打印 Duke 在指定轮次（默认 E8/F4/Final）的赛事信息"""
    print("=" * 70)
    print("【3】Duke 赛事信息（重点：E8 / F4 / Final）")
    print("=" * 70)
    by_round = {}
    for g in duke_games:
        by_round.setdefault(g["round"], []).append(g)
    for r in round_order:
        games = by_round.get(r, [])
        rname = ROUND_NAMES.get(r, f"Round{r}")
        print(f"\n  >> {rname} (round={r}) 共 {len(games)} 场可能对阵")
        print("  " + "-" * 66)
        for g in sorted(games, key=lambda x: -x["duke_win_sub"]):
            opp = g["opp_name"]
            base_s = f"{g['duke_win_base']:.2%}" if g["duke_win_base"] is not None else "N/A"
            sub_s = f"{g['duke_win_sub']:.2%}"
            diff = ""
            if g["duke_win_base"] is not None:
                d = g["duke_win_sub"] - g["duke_win_base"]
                diff = f" (diff={d:+.2%})" if abs(d) > 0.001 else ""
            print(f"    Duke vs {opp:20}  base={base_s:>6}  sub={sub_s:>6}{diff}")
    print()


def sampling_check(mens_seeds, sub_duke, base_map):
    """抽样检查：若干关键 ID 的 base vs submission 是否合理"""
    print("=" * 70)
    print("【4】抽样检查（关键对阵）")
    print("=" * 70)
    # ID 格式：2026_较小TeamID_较大TeamID
    samples = [
        ("2026_1181_1276", "Duke vs Michigan (Final)"),
        ("2026_1112_1181", "Duke vs Arizona (Final)"),
        ("2026_1112_1276", "Michigan vs Arizona (F4)"),
        ("2026_1181_1403", "Texas Tech vs Duke"),
        ("2026_1276_1403", "Texas Tech vs Michigan"),
        ("2026_1103_1181", "Akron vs Duke (不同区->F4/Final)"),
        ("2026_1181_1202", "Duke vs Furman (弱队)"),
    ]
    for uid, desc in samples:
        pred_sub = sub_duke.get(uid)
        pred_base = base_map.get(uid)
        if pred_sub is None:
            print(f"  {uid} {desc}: submission 中无此 ID")
        else:
            base_s = f"{pred_base:.4f}" if pred_base is not None else "N/A"
            print(f"  {uid} {desc}")
            print(f"       base={base_s}  sub={pred_sub:.4f}")
    print()


def verify_texas_tech_coverage(tt_adj, id2name, mp_men):
    """校验 Texas Tech 伤病表与 matchupProbabilities_men 队名是否匹配"""
    print("=" * 70)
    print("【5】Texas Tech 伤病表 vs 队名匹配校验")
    print("=" * 70)
    mp_names = set(id2name.values())
    tt_names = set(tt_adj.keys())
    missing_in_mp = tt_names - mp_names
    missing_in_tt = mp_names - tt_names - {"Texas Tech"}
    if missing_in_mp:
        print(f"  [WARN] 伤病表有但 mp 无的队名: {list(missing_in_mp)[:5]}...")
    else:
        print("  伤病表 OppName 与 matchupProbabilities_men 队名完全匹配 [OK]")
    print(f"  伤病表对手数: {len(tt_adj)}, mp 队数: {len(mp_names)}")
    print()


def main():
    data_dir = os.path.join(ROOT, "march-machine-learning-mania-2026")
    mens_seeds = pd.read_csv(os.path.join(data_dir, "MNCAATourneySeeds.csv"))
    mens_seeds = mens_seeds[mens_seeds["Season"] == 2026].copy()

    verify_seeds_and_ids(mens_seeds)
    if not verify_file_paths():
        print("存在缺失文件，停止验证")
        return

    # 加载数据
    mp_men = pd.read_csv(os.path.join(ROOT, "matchupProbabilities_men.csv"))
    base_full = pd.read_csv(os.path.join(ROOT, "matchupProbabilities.csv"))
    if "ID" not in base_full.columns and base_full.shape[1] >= 2:
        base_full = base_full.rename(columns={base_full.columns[1]: "ID", base_full.columns[2]: "Pred"})
    base_map = dict(zip(base_full["ID"].astype(str), base_full["Pred"].astype(float)))

    sub_duke = pd.read_csv(os.path.join(ROOT, "outputs", "submission_A_Duke.csv"))
    sub_map = dict(zip(sub_duke["ID"].astype(str), sub_duke["Pred"].astype(float)))

    id2name = {}
    for _, r in mp_men.iterrows():
        id2name[r["T1_TeamID"]] = r["T1_TeamName"]
        id2name[r["T2_TeamID"]] = r["T2_TeamName"]

    tt_path = os.path.join(ROOT, "docs", "Texas_Tech_对阵预测_伤病下调版.csv")
    tt_df = pd.read_csv(tt_path)
    tt_adj = dict(zip(tt_df["OppName"], tt_df["TT_WinProb_下调后"]))

    duke_games = build_duke_matchups_by_round(mens_seeds, id2name, base_map, sub_map)
    print_duke_tournament_path(duke_games)

    sampling_check(mens_seeds, sub_map, base_map)
    verify_texas_tech_coverage(tt_adj, id2name, mp_men)

    print("=" * 70)
    print("【6】验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
