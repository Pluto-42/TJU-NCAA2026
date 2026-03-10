"""
手动覆盖策略模块 (Manual Override Strategy)

基于 Brier Score 下 33.3% 最优押注策略，实现对指定风险队在其可能晋级轮次内的 Pred 覆盖。
参考: updated-goto-conversion-winning-solution, ncaa2025-3th/7th-solution
"""

import pandas as pd


def get_roundOfMatch(team1: int, team2: int, seeds_df: pd.DataFrame) -> int:
    """
    根据 bracket 结构推断 team1 vs team2 对阵出现的轮次。

    Returns:
        1: First Four (play-in)
        2: Round of 64
        3: Round of 32
        4: Sweet Sixteen
        5: Elite Eight
        6: Final Four (半决赛)
        7: National Final (决赛)
    """
    slotMap = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    team1_seed = seeds_df.loc[seeds_df["TeamID"] == team1, "Seed"].values[0]
    team2_seed = seeds_df.loc[seeds_df["TeamID"] == team2, "Seed"].values[0]

    # First Four: 同赛区同种子 (如 W16a vs W16b)
    isFirstFourMatch = team1_seed[:3] == team2_seed[:3]
    if isFirstFourMatch:
        return 1

    team1_region = str(team1_seed[0])
    team2_region = str(team2_seed[0])
    team1_seedNumber = int(team1_seed[1:3].rstrip("ab"))
    team2_seedNumber = int(team2_seed[1:3].rstrip("ab"))

    isRegionSame = team1_region == team2_region
    if not isRegionSame:
        isTeam1_regionWX = team1_region in ["W", "X"]
        isTeam2_regionWX = team2_region in ["W", "X"]
        if isTeam1_regionWX and isTeam2_regionWX:
            return 6
        if (not isTeam1_regionWX) and (not isTeam2_regionWX):
            return 6
        return 7

    # Same region
    try:
        team1_slot = slotMap.index(team1_seedNumber)
        team2_slot = slotMap.index(team2_seedNumber)
    except ValueError:
        return 2  # fallback for play-in seeds

    isRound2 = (team1_slot // 2) == (team2_slot // 2)
    if isRound2:
        return 2
    isRound3 = (team1_slot // 4) == (team2_slot // 4)
    if isRound3:
        return 3
    isRound4 = (team1_slot // 8) == (team2_slot // 8)
    if isRound4:
        return 4
    return 5


def get_tourneyFlag(team1: int, team2: int, seeds_df: pd.DataFrame) -> int:
    """
    若两队均在锦标赛内，返回对阵轮次；否则返回 0。
    """
    tourneyTeams = seeds_df["TeamID"].tolist()
    if team1 in tourneyTeams and team2 in tourneyTeams:
        return get_roundOfMatch(team1, team2, seeds_df)
    return 0


def get_flag_list(
    submission_df: pd.DataFrame,
    mens_seeds_df: pd.DataFrame,
    womens_seeds_df: pd.DataFrame,
    id_col: str = "ID",
) -> list:
    """
    对 submission 每一行计算对阵轮次 flag。
    team1 + team2 > 6000 判为女篮。
    """
    flag_list = []
    for i in range(len(submission_df)):
        row = submission_df.iloc[i]
        id_val = row[id_col] if id_col in row.index else row.iloc[0]
        parts = str(id_val).split("_")
        team1, team2 = int(parts[1]), int(parts[2])
        is_womens = team1 + team2 > 6000
        seeds_df = womens_seeds_df if is_womens else mens_seeds_df
        flag_list.append(get_tourneyFlag(team1, team2, seeds_df))
    return flag_list


def set_optimalStrategy(
    submission_df: pd.DataFrame,
    mens_seeds_df: pd.DataFrame,
    womens_seeds_df: pd.DataFrame,
    risk_teams,
    risk_team_to_win_rounds,
    id_col: str = "ID",
    pred_col: str = "Pred",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    对指定风险队在其可能晋级的前若干轮内，强制 Pred=1（若风险队为 team1）或 Pred=0（若风险队为 team2）。

    Args:
        submission_df: 含 ID, Pred 的 DataFrame
        mens_seeds_df: 男篮 seeds (Season, TeamID, Seed)
        womens_seeds_df: 女篮 seeds
        risk_teams: 风险队 TeamID 列表，如 [1179] 或 [1181, 1120, ...]
        risk_team_to_win_rounds: 对应每队押其赢到的轮次，如 [2] 或 [6,5,5,5,6,6]
        id_col, pred_col: 列名
        verbose: 是否打印被覆盖的行

    Returns:
        修改后的 submission_df
    """
    risk_teams = list(risk_teams)
    risk_team_to_win_rounds = list(risk_team_to_win_rounds)
    if len(risk_teams) != len(risk_team_to_win_rounds):
        raise ValueError("risk_teams 与 risk_team_to_win_rounds 长度必须相同")

    risk_dict = dict(zip(risk_teams, risk_team_to_win_rounds))
    flag_list = get_flag_list(submission_df, mens_seeds_df, womens_seeds_df, id_col)

    for i in range(len(submission_df)):
        row = submission_df.iloc[i]
        id_val = row[id_col] if id_col in submission_df.columns else submission_df.iloc[i, 0]
        parts = str(id_val).split("_")
        team1, team2 = int(parts[1]), int(parts[2])
        submission_round = flag_list[i]

        if team1 in risk_dict and 0 < submission_round <= risk_dict[team1]:
            submission_df.at[submission_df.index[i], pred_col] = 1.0
            if verbose:
                print(submission_df.iloc[i])
        elif team2 in risk_dict and 0 < submission_round <= risk_dict[team2]:
            submission_df.at[submission_df.index[i], pred_col] = 0.0
            if verbose:
                print(submission_df.iloc[i])

    return submission_df


def apply_manual_overrides(
    submission_df: pd.DataFrame,
    overrides: dict,
    id_col: str = "ID",
    pred_col: str = "Pred",
) -> pd.DataFrame:
    """
    对指定 ID 的行应用固定 Pred 覆盖。

    Args:
        submission_df: 含 ID, Pred
        overrides: {"2025_1124_1280": 0.9818, ...}

    Returns:
        修改后的 submission_df
    """
    for i in range(len(submission_df)):
        id_val = submission_df.iloc[i][id_col]
        if id_val in overrides:
            submission_df.at[submission_df.index[i], pred_col] = overrides[id_val]
    return submission_df
