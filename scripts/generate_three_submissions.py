#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三套提交生成脚本：A(Duke) / B(Michigan) / C(Arizona)

基于 matchupProbabilities_men.csv，在 Texas Tech 伤病下调覆写基础上，
对每条冠军主线按「前期少调、后期猛调」策略做激进押注，生成三份 submission 文件。

逻辑顺序：
  1. 从 matchupProbabilities_men 填入男篮预测，女篮保持 0.5
  2. 对涉及 Texas Tech 的场次：用伤病下调版胜率覆写（强队-7%/中强-4%/中弱-2.5%）
  3. 对冠军主线相关场次：按轮次策略调整（R64/R32 少调，S16 +8%，E8/F4/Final 猛拉到 0.85–0.92）

参考：docs/三套提交_调整幅度建议.md
"""

import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.optimal_strategy import get_tourneyFlag

# --- 常量定义 ---
# 冠军主线 TeamID：Duke=1181, Michigan=1276, Arizona=1112
CHAMPION_IDS = [
    ("A_Duke", 1181),
    ("B_Michigan", 1276),
    ("C_Arizona", 1112),
]
# Texas Tech 伤病下调
TT_TEAM_ID = 1403
# 轮次映射：get_roundOfMatch 返回 1(play-in), 2(R64), 3(R32), 4(S16), 5(E8), 6(F4), 7(Final)
# 策略：2,3 少调；4 S16 +8%；5,6,7 猛调


def load_base_predictions(season: int = 2026) -> pd.DataFrame:
    """
    加载基础预测：从 matchupProbabilities_men 与 SampleSubmissionStage2 合并。
    - 男篮对阵：使用 matchupProbabilities_men 的 Pred
    - 女篮及其他：保持 0.5
    """
    mp_path = os.path.join(ROOT, "matchupProbabilities_men.csv")
    sample_path = os.path.join(ROOT, "march-machine-learning-mania-2026", "SampleSubmissionStage2.csv")

    mp = pd.read_csv(mp_path)
    sample = pd.read_csv(sample_path)

    # 建立 ID -> Pred 映射（男篮）
    mp_dict = dict(zip(mp["ID"].astype(str), mp["Pred"].astype(float)))

    # 对 sample 每行：若 ID 在 mp 中则用 mp 的 Pred，否则保持 0.5
    preds = []
    for _, row in sample.iterrows():
        id_val = str(row["ID"])
        preds.append(mp_dict.get(id_val, 0.5))

    df = sample.copy()
    df["Pred"] = preds
    return df


def load_texas_tech_injury_overrides() -> dict:
    """
    加载 Texas Tech 伤病下调表：OppName -> TT 胜率（下调后）。
    用于覆写所有涉及 Texas Tech 的对阵。
    """
    tt_path = os.path.join(ROOT, "docs", "Texas_Tech_对阵预测_伤病下调版.csv")
    tt_df = pd.read_csv(tt_path)
    # OppName -> TT 胜率
    return dict(zip(tt_df["OppName"], tt_df["TT_WinProb_下调后"]))


def load_team_id_to_name() -> dict:
    """从 matchupProbabilities_men 建立 TeamID -> TeamName 映射（用于 Texas Tech 对手名查找）"""
    mp_path = os.path.join(ROOT, "matchupProbabilities_men.csv")
    mp = pd.read_csv(mp_path)
    id2name = {}
    for _, r in mp.iterrows():
        id2name[r["T1_TeamID"]] = r["T1_TeamName"]
        id2name[r["T2_TeamID"]] = r["T2_TeamName"]
    return id2name


def apply_texas_tech_overrides(
    df: pd.DataFrame,
    tt_adj: dict,
    id2name: dict,
) -> pd.DataFrame:
    """
    对涉及 Texas Tech 的对阵应用伤病下调覆写。
    Pred 表示 T1 胜率。若 TT 为 T1 则 Pred=TT_win_prob；若 TT 为 T2 则 Pred=1-TT_win_prob。
    """
    df = df.copy()
    for i in range(len(df)):
        parts = str(df.iloc[i]["ID"]).split("_")
        if len(parts) != 3:
            continue
        t1, t2 = int(parts[1]), int(parts[2])
        if t1 != TT_TEAM_ID and t2 != TT_TEAM_ID:
            continue
        opp_id = t2 if t1 == TT_TEAM_ID else t1
        opp_name = id2name.get(opp_id)
        if opp_name is None or opp_name not in tt_adj:
            continue
        tt_win = tt_adj[opp_name]
        if t1 == TT_TEAM_ID:
            df.at[df.index[i], "Pred"] = tt_win
        else:
            df.at[df.index[i], "Pred"] = 1.0 - tt_win
    return df


def apply_champion_aggressive_overrides(
    df: pd.DataFrame,
    champion_id: int,
    mens_seeds: pd.DataFrame,
    womens_seeds: pd.DataFrame,
) -> pd.DataFrame:
    """
    对冠军主线相关对阵应用「前期少调、后期猛调」策略。
    - round 2, 3：微调或不动（pred>0.88 不调；0.75–0.88 最多 +2%）
    - round 4：+8%
    - round 5, 6, 7：pred = max(pred+0.20, 0.85)，cap 0.92
    """
    df = df.copy()
    # 只处理男篮行
    for i in range(len(df)):
        parts = str(df.iloc[i]["ID"]).split("_")
        if len(parts) != 3:
            continue
        t1, t2 = int(parts[1]), int(parts[2])
        if t1 + t2 > 6000:
            continue
        if champion_id not in (t1, t2):
            continue
        rnd = get_tourneyFlag(t1, t2, mens_seeds)
        if rnd == 0:
            continue
        pred = float(df.iloc[i]["Pred"])
        champ_is_t1 = t1 == champion_id
        if not champ_is_t1:
            pred = 1.0 - pred
        # 应用轮次策略
        if rnd in (2, 3):
            if pred <= 0.75:
                pred = min(pred + 0.02, 0.88)
            elif pred < 0.88:
                pred = min(pred + 0.01, 0.88)
        elif rnd == 4:
            pred = min(pred + 0.08, 0.98)
        elif rnd in (5, 6, 7):
            pred = min(max(pred + 0.20, 0.85), 0.92)
        if not champ_is_t1:
            pred = 1.0 - pred
        df.at[df.index[i], "Pred"] = pred
    return df


def main():
    data_dir = os.path.join(ROOT, "march-machine-learning-mania-2026")
    seeds_path = os.path.join(data_dir, "MNCAATourneySeeds.csv")
    if not os.path.exists(seeds_path):
        raise FileNotFoundError(f"MNCAATourneySeeds.csv 不存在: {seeds_path}")

    mens_seeds = pd.read_csv(seeds_path)
    mens_seeds = mens_seeds[mens_seeds["Season"] == 2026].copy()
    if len(mens_seeds) == 0:
        raise ValueError("2026 赛季 seeds 为空，请检查 MNCAATourneySeeds.csv")

    # 断言：Duke/Michigan/Arizona/Texas Tech 必须在 2026 seeds 中
    required_ids = {1181, 1276, 1112, 1403}
    seed_ids = set(mens_seeds["TeamID"].tolist())
    missing = required_ids - seed_ids
    if missing:
        raise ValueError(f"2026 seeds 中缺少关键 TeamID: {missing}")
    womens_seeds = pd.read_csv(
        os.path.join(data_dir, "WNCAATourneySeeds.csv"),
    )
    womens_seeds = womens_seeds[womens_seeds["Season"] == 2026].copy()

    # 1. 加载基础预测
    print("加载基础预测...")
    df = load_base_predictions(2026)
    print(f"  总行数: {len(df)}")

    # 2. 加载 Texas Tech 伤病下调
    tt_adj = load_texas_tech_injury_overrides()
    id2name = load_team_id_to_name()
    print(f"  Texas Tech 对手覆写数: {len(tt_adj)}")

    # 3. 先生成一份「基础 + Texas Tech 覆写」的通用底稿
    df_base = apply_texas_tech_overrides(df, tt_adj, id2name)

    # 4. 对每条冠军主线生成一份提交
    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # A=Duke, B=Michigan, C=Arizona
    for label, cid in CHAMPION_IDS:
        df_out = apply_champion_aggressive_overrides(
            df_base.copy(),
            cid,
            mens_seeds,
            womens_seeds,
        )
        out_path = os.path.join(out_dir, f"submission_{label}.csv")
        df_out[["ID", "Pred"]].to_csv(out_path, index=False)
        print(f"  已写入: {out_path}")

    print("\n完成。三套提交：A(Duke), B(Michigan), C(Arizona)")


if __name__ == "__main__":
    main()
