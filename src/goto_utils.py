"""
goto_conversion 工具模块

用于加载由 goto_conversion 生成的概率表，并将其用于 submission 或特征。
概率表来源: https://github.com/gotoConversion/goto_conversion
"""

import pandas as pd
import os
from typing import Optional, Tuple


def load_probability_tables(
    data_dir: str,
    mens_file: str = "mensProbabilitiesTable.csv",
    womens_file: str = "womensProbabilitiesTable.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载男女篮概率表（CSV 格式，列 player/Team, rd1_win~rd6_win）。

    Args:
        data_dir: 目录路径（如 538data 或 goto_conversion/outputFiles）
        mens_file, womens_file: 文件名

    Returns:
        (mensProbabilities_df, womensProbabilities_df)，index 为队伍名
    """
    mens_path = os.path.join(data_dir, mens_file)
    womens_path = os.path.join(data_dir, womens_file)
    mens_df = pd.read_csv(mens_path, index_col="player")
    womens_df = pd.read_csv(womens_path, index_col="player")
    if "Elo_Rating" in mens_df.columns:
        mens_df = mens_df.drop("Elo_Rating", axis=1)
    if "Elo_Rating" in womens_df.columns:
        womens_df = womens_df.drop("Elo_Rating", axis=1)
    return mens_df, womens_df


def load_probability_table_from_xlsx(
    filepath: str,
    index_col: str = "Team",
) -> pd.DataFrame:
    """
    加载 Substack 下载的 XLSX 概率表。

    Args:
        filepath: XLSX 文件路径
        index_col: 作为 index 的列名（Substack 通常为 "Team"）

    Returns:
        含 rd1_win~rd6_win 的 DataFrame，index 为队伍名
    """
    df = pd.read_excel(filepath)
    if index_col in df.columns:
        df = df.set_index(index_col)
    return df


def get_match_prob_from_table(
    team1_name: str,
    team2_name: str,
    prob_df: pd.DataFrame,
) -> Optional[float]:
    """
    从概率表中查 team1 对 team2 的胜率。
    注意：prob_df 格式需为 (player, rd1_win, rd2_win, ...)，本函数已弃用 pairwise 矩阵假设。
    实际应使用 get_pairwise_prob_from_rd_win()。
    """
    if team1_name not in prob_df.index or team2_name not in prob_df.index:
        return None
    return prob_df.loc[team1_name, team2_name]


def get_pairwise_prob_from_rd_win(
    team1_name: str,
    team2_name: str,
    prob_df: pd.DataFrame,
    round_num: int,
) -> Optional[float]:
    """
    从 rd*_win 格式的概率表计算 team1 对 team2 的胜率。

    使用 Bradley-Terry 近似：
    P(team1 胜) = rd_r_win(team1) / [rd_r_win(team1) + rd_r_win(team2)]

    Args:
        team1_name, team2_name: 表内 player/index 名称
        prob_df: 含 rd1_win, rd2_win, ... 列的 DataFrame，index 为队伍名
        round_num: 相遇轮次 1~6

    Returns:
        team1 的胜率，若任一队不在表中则返回 None
    """
    col = f"rd{round_num}_win"
    if col not in prob_df.columns:
        return None
    if team1_name not in prob_df.index or team2_name not in prob_df.index:
        return None
    r1 = prob_df.loc[team1_name, col]
    r2 = prob_df.loc[team2_name, col]
    total = r1 + r2
    if total <= 0:
        return 0.5
    return float(r1 / total)
