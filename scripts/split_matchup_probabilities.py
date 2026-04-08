"""
将 matchupProbabilities.csv 拆分为男女篮两张表，
仅保留双方均在锦标赛 seeds 中的对阵，并添加两队队名列。
"""

import pandas as pd
from pathlib import Path

# 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "march-machine-learning-mania-2026"
INPUT_CSV = ROOT / "matchupProbabilities.csv"
OUTPUT_MEN = ROOT / "matchupProbabilities_men.csv"
OUTPUT_WOMEN = ROOT / "matchupProbabilities_women.csv"

SEASON = 2026


def main():
    # 1. 加载 matchupProbabilities
    df = pd.read_csv(INPUT_CSV)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["T1_TeamID"] = df["ID"].str.split("_").str[1].astype(int)
    df["T2_TeamID"] = df["ID"].str.split("_").str[2].astype(int)

    # 2. 加载 2026 seeds
    m_seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    w_seeds = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")
    men_tourney = set(
        m_seeds.loc[m_seeds["Season"] == SEASON, "TeamID"].astype(int).tolist()
    )
    women_tourney = set(
        w_seeds.loc[w_seeds["Season"] == SEASON, "TeamID"].astype(int).tolist()
    )

    # 3. 加载队名
    m_teams = pd.read_csv(DATA_DIR / "MTeams.csv")[["TeamID", "TeamName"]]
    w_teams = pd.read_csv(DATA_DIR / "WTeams.csv")[["TeamID", "TeamName"]]
    m_names = dict(zip(m_teams["TeamID"], m_teams["TeamName"]))
    w_names = dict(zip(w_teams["TeamID"], w_teams["TeamName"]))

    # 4. 男篮：双方均在 men_tourney
    men_mask = df["T1_TeamID"].isin(men_tourney) & df["T2_TeamID"].isin(men_tourney)
    df_men = df.loc[men_mask].copy()
    df_men["T1_TeamName"] = df_men["T1_TeamID"].map(m_names)
    df_men["T2_TeamName"] = df_men["T2_TeamID"].map(m_names)
    df_men = df_men[["ID", "T1_TeamID", "T1_TeamName", "T2_TeamID", "T2_TeamName", "Pred"]]
    df_men.to_csv(OUTPUT_MEN, index=False)
    print(f"男篮锦标赛对阵: {len(df_men)} 行 -> {OUTPUT_MEN}")

    # 5. 女篮：双方均在 women_tourney
    women_mask = df["T1_TeamID"].isin(women_tourney) & df["T2_TeamID"].isin(women_tourney)
    df_women = df.loc[women_mask].copy()
    df_women["T1_TeamName"] = df_women["T1_TeamID"].map(w_names)
    df_women["T2_TeamName"] = df_women["T2_TeamID"].map(w_names)
    df_women = df_women[["ID", "T1_TeamID", "T1_TeamName", "T2_TeamID", "T2_TeamName", "Pred"]]
    df_women.to_csv(OUTPUT_WOMEN, index=False)
    print(f"女篮锦标赛对阵: {len(df_women)} 行 -> {OUTPUT_WOMEN}")


if __name__ == "__main__":
    main()
