"""
从 pairwise matchupProbabilities 推导队伍实力排名，
并可对照 projection 表（rd6_win / Elo）作为自上而下的参考。

方法：
-  pairwise 排名：每队「预期胜场」= sum over 所有对手 of P(该队胜)
   - 当该队为 T1 时用 Pred，为 T2 时用 1-Pred
   - 预期胜场越高，实力越强
- projection 排名：按 rd6_win（夺冠概率）或 Elo_Rating 降序
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEN_PAIRWISE = ROOT / "matchupProbabilities_men.csv"
WOMEN_PAIRWISE = ROOT / "matchupProbabilities_women.csv"
# projection 表路径（可按需改为 2026 文件）
MEN_PROJ = ROOT / "goto_conversion-main/goto_conversion/outputFiles/mensProbabilitiesTable2025.csv"
WOMEN_PROJ = ROOT / "goto_conversion-main/goto_conversion/outputFiles/womensProbabilitiesTable2025.csv"
OUT_DIR = ROOT / "docs"  # 输出到 docs 便于查看


def rank_from_pairwise(csv_path: Path) -> pd.DataFrame:
    """从 pairwise 表计算每队预期胜场并排名。"""
    df = pd.read_csv(csv_path)
    teams = set(df["T1_TeamID"].tolist()) | set(df["T2_TeamID"].tolist())
    names = dict(zip(df["T1_TeamID"], df["T1_TeamName"]))
    names.update(zip(df["T2_TeamID"], df["T2_TeamName"]))

    expected_wins = {}
    for t in teams:
        wins = 0.0
        # T 为 T1 的对阵
        mask1 = df["T1_TeamID"] == t
        wins += df.loc[mask1, "Pred"].sum()
        # T 为 T2 的对阵
        mask2 = df["T2_TeamID"] == t
        wins += (1 - df.loc[mask2, "Pred"]).sum()
        expected_wins[t] = wins

    rows = [
        {
            "TeamID": t,
            "TeamName": names.get(t, str(t)),
            "ExpectedWins": round(expected_wins[t], 2),
        }
        for t in teams
    ]
    res = pd.DataFrame(rows).sort_values("ExpectedWins", ascending=False).reset_index(drop=True)
    res["PairwiseRank"] = res.index + 1
    return res[["PairwiseRank", "TeamID", "TeamName", "ExpectedWins"]]


def rank_from_projection(csv_path: Path) -> pd.DataFrame:
    """从 projection 表按 rd6_win 排名（排除占位符如 Y16、W16）。"""
    df = pd.read_csv(csv_path)
    # 排除 bracket 占位符（通常为 区域+数字 如 Y16, W16, X11）
    is_placeholder = df["player"].str.match(r"^[WXYZ]\d{2}[ab]?$", case=False, na=False)
    real = df.loc[~is_placeholder].copy()
    real = real.groupby("player", as_index=False).agg(
        {"rd6_win": "max", "Elo_Rating": "first"}
    )
    real = real.sort_values("rd6_win", ascending=False).reset_index(drop=True)
    real["ProjRank"] = real.index + 1
    real = real.rename(columns={"player": "TeamName"})
    return real[["ProjRank", "TeamName", "rd6_win", "Elo_Rating"]]


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)

    # 1. 男篮 pairwise 排名
    men_pairwise = rank_from_pairwise(MEN_PAIRWISE)
    men_pairwise.to_csv(out_dir / "team_ranking_men_pairwise.csv", index=False)
    print("男篮 pairwise 排名已写入:", out_dir / "team_ranking_men_pairwise.csv")
    print(men_pairwise.head(15).to_string(index=False))
    print()

    # 2. 女篮 pairwise 排名
    women_pairwise = rank_from_pairwise(WOMEN_PAIRWISE)
    women_pairwise.to_csv(out_dir / "team_ranking_women_pairwise.csv", index=False)
    print("女篮 pairwise 排名已写入:", out_dir / "team_ranking_women_pairwise.csv")
    print(women_pairwise.head(15).to_string(index=False))
    print()

    # 3. Projection 表排名（若有）
    if MEN_PROJ.exists():
        men_proj = rank_from_projection(MEN_PROJ)
        men_proj.to_csv(out_dir / "team_ranking_men_projection.csv", index=False)
        print("男篮 projection 排名已写入:", out_dir / "team_ranking_men_projection.csv")
        print(men_proj.head(15).to_string(index=False))
    if WOMEN_PROJ.exists():
        women_proj = rank_from_projection(WOMEN_PROJ)
        women_proj.to_csv(out_dir / "team_ranking_women_projection.csv", index=False)
        print("女篮 projection 排名已写入:", out_dir / "team_ranking_women_projection.csv")
        print(women_proj.head(15).to_string(index=False))

    print("\n完成。pairwise 排名来自 matchupProbabilities (2026)；projection 来自 2025 表，可替换为 2026 版本。")


if __name__ == "__main__":
    main()
