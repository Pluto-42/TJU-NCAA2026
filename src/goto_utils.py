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
    加载男女篮概率表。

    Args:
        data_dir: 目录路径（如 538data 或 goto_conversion/outputFiles）
        mens_file, womens_file: 文件名

    Returns:
        (mensProbabilities_df, womensProbabilities_df)
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


def get_match_prob_from_table(
    team1_name: str,
    team2_name: str,
    prob_df: pd.DataFrame,
) -> Optional[float]:
    """
    从概率表中查 team1 对 team2 的胜率。
    需配合 MTeamSpellings/WTeamSpellings 将 TeamID 映射为表内 player 名称。
    """
    if team1_name not in prob_df.index or team2_name not in prob_df.index:
        return None
    return prob_df.loc[team1_name, team2_name]
