"""
Raddar 框架工具函数 (基于 vilnius-ncaa.ipynb)

包含 prepare_data 及常用特征工程辅助函数。
"""

import pandas as pd
import numpy as np


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 Detailed Results 做对称化处理：W/L 对调翻倍，消除位置偏差。

    - 加时赛统计按 (40+5*NumOT)/40 归一化
    - 输出列: Season, DayNum, T1_TeamID, T2_TeamID, T1_Score, T2_Score, ... PointDiff, win, men_women
    """
    df = df[
        [
            "Season",
            "DayNum",
            "LTeamID",
            "LScore",
            "WTeamID",
            "WScore",
            "NumOT",
            "LFGM",
            "LFGA",
            "LFGM3",
            "LFGA3",
            "LFTM",
            "LFTA",
            "LOR",
            "LDR",
            "LAst",
            "LTO",
            "LStl",
            "LBlk",
            "LPF",
            "WFGM",
            "WFGA",
            "WFGM3",
            "WFGA3",
            "WFTM",
            "WFTA",
            "WOR",
            "WDR",
            "WAst",
            "WTO",
            "WStl",
            "WBlk",
            "WPF",
        ]
    ].copy()

    adjot = (40 + 5 * df["NumOT"]) / 40
    adjcols = [
        "LScore",
        "WScore",
        "LFGM",
        "LFGA",
        "LFGM3",
        "LFGA3",
        "LFTM",
        "LFTA",
        "LOR",
        "LDR",
        "LAst",
        "LTO",
        "LStl",
        "LBlk",
        "LPF",
        "WFGM",
        "WFGA",
        "WFGM3",
        "WFGA3",
        "WFTM",
        "WFTA",
        "WOR",
        "WDR",
        "WAst",
        "WTO",
        "WStl",
        "WBlk",
        "WPF",
    ]
    for col in adjcols:
        df[col] = df[col] / adjot

    dfswap = df.copy()
    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in df.columns]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in dfswap.columns]
    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output["PointDiff"] = output["T1_Score"] - output["T2_Score"]
    output["win"] = (output["PointDiff"] > 0).astype(int)
    output["men_women"] = output["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)
    return output


def extract_seed_number(seed_str: str) -> int:
    """
    从 Seed 字符串提取数值，如 W01 -> 1, Y16a -> 16。
    """
    return int(seed_str[1:3].rstrip("ab"))
