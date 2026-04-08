"""
Raddar 流水线 (完整复刻 vilnius-ncaa.ipynb 核心逻辑)

流程：
1. 加载 M/W 常规赛 + 锦标赛 Detailed Results
2. prepare_data 对称化 (W/L 翻倍、加时归一)
3. 构建 boxcols、ss、ss_T1、ss_T2
4. Seeds、Elo、GLM quality 特征
5. Leave-one-season-out XGBoost + Spline 校准
6. 从 SampleSubmissionStage2 构造 X，预测并输出 submission
"""

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss
from tqdm import tqdm

from src.raddar_utils import extract_seed_number, prepare_data

warnings.filterwarnings("ignore")

# ============ 常量 (与 vilnius 一致) ============
BOXCOLS = [
    "T1_Score", "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3", "T1_FTM", "T1_FTA",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_Score", "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3", "T2_FTM", "T2_FTA",
    "T2_OR", "T2_DR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk", "T2_PF",
    "PointDiff",
]
FEATURES = [
    "men_women", "T1_seed", "T2_seed", "Seed_diff",
    "T1_avg_Score", "T1_avg_FGA", "T1_avg_Blk", "T1_avg_PF",
    "T1_avg_opponent_FGA", "T1_avg_opponent_Blk", "T1_avg_opponent_PF",
    "T1_avg_PointDiff",
    "T2_avg_Score", "T2_avg_FGA", "T2_avg_Blk", "T2_avg_PF",
    "T2_avg_opponent_FGA", "T2_avg_opponent_Blk", "T2_avg_opponent_PF",
    "T2_avg_PointDiff",
    "T1_elo", "T2_elo", "elo_diff",
    "T1_quality", "T2_quality",
]
BASE_ELO = 1000
ELO_WIDTH = 400
K_FACTOR = 100
SEASON_MIN = 2003
SPLINE_T = 25
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.01,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 4,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 32,
}
NUM_ROUNDS = 700


def _load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载常规赛、锦标赛、Seeds 数据并合并男女篮。"""
    data_dir = Path(data_dir)
    M_regular = pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv")
    W_regular = pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv")
    M_tourney = pd.read_csv(data_dir / "MNCAATourneyDetailedResults.csv")
    W_tourney = pd.read_csv(data_dir / "WNCAATourneyDetailedResults.csv")
    M_seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    W_seeds = pd.read_csv(data_dir / "WNCAATourneySeeds.csv")

    regular_results = pd.concat([M_regular, W_regular], ignore_index=True)
    tourney_results = pd.concat([M_tourney, W_tourney], ignore_index=True)
    seeds = pd.concat([M_seeds, W_seeds], ignore_index=True)

    regular_results = regular_results.loc[regular_results["Season"] >= SEASON_MIN]
    tourney_results = tourney_results.loc[tourney_results["Season"] >= SEASON_MIN]
    seeds = seeds.loc[seeds["Season"] >= SEASON_MIN]
    return regular_results, tourney_results, seeds


def _compute_elo(regular_data: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """按赛季计算 Elo，仅对 win==1 的行更新（T1 胜）。"""
    elos = []
    for season in sorted(seeds["Season"].unique()):
        ss = regular_data.loc[(regular_data["Season"] == season) & (regular_data["win"] == 1)].reset_index(drop=True)
        teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
        elo = {t: BASE_ELO for t in teams}

        def expected_result(elo_a, elo_b):
            return 1.0 / (1 + 10 ** ((elo_b - elo_a) / ELO_WIDTH))

        def update_elo(winner_elo, loser_elo):
            exp = expected_result(winner_elo, loser_elo)
            change = K_FACTOR * (1 - exp)
            return winner_elo + change, loser_elo - change

        for i in range(ss.shape[0]):
            w_team = ss.loc[i, "T1_TeamID"]
            l_team = ss.loc[i, "T2_TeamID"]
            w_elo, l_elo = elo[w_team], elo[l_team]
            elo[w_team], elo[l_team] = update_elo(w_elo, l_elo)

        elo_df = pd.DataFrame(list(elo.items()), columns=["TeamID", "elo"])
        elo_df["Season"] = season
        elos.append(elo_df)
    return pd.concat(elos, ignore_index=True)


# Glicko 常量：q=ln(10)/400；c 控制 RD 随时间增长，game-by-game 取较小值
GLICKO_Q = math.log(10) / 400
GLICKO_C = 35
GLICKO_RD_MIN = 30
GLICKO_RD_MAX = 350
GLICKO_INIT_R = 1500
GLICKO_INIT_RD = 350


def _glicko_g(rd: float) -> float:
    """Glicko g(RD) 函数：压缩对手 RD 对预期分数的影响。"""
    return 1.0 / math.sqrt(1.0 + 3.0 * (GLICKO_Q**2) * (rd**2) / (math.pi**2))


def _compute_glicko(regular_data: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """
    按赛季计算 Glicko 评分，逐场更新。
    返回格式与 _compute_elo 一致：(TeamID, elo, Season)，elo 列存 Glicko rating。
    """
    glickos = []
    for season in sorted(seeds["Season"].unique()):
        ss = regular_data.loc[(regular_data["Season"] == season) & (regular_data["win"] == 1)].reset_index(
            drop=True
        )
        teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
        r = {t: GLICKO_INIT_R for t in teams}
        rd = {t: GLICKO_INIT_RD for t in teams}

        for i in range(ss.shape[0]):
            w_team = ss.loc[i, "T1_TeamID"]
            l_team = ss.loc[i, "T2_TeamID"]
            rw, rl = r[w_team], r[l_team]
            rdw, rdl = rd[w_team], rd[l_team]

            # Step 1: 赛前 RD 增长（时间带来的不确定性）
            rdw = min(math.sqrt(rdw**2 + GLICKO_C**2), GLICKO_RD_MAX)
            rdl = min(math.sqrt(rdl**2 + GLICKO_C**2), GLICKO_RD_MAX)

            # Step 2: 胜方更新（对败方）
            g_l = _glicko_g(rdl)
            e_w = 1.0 / (1.0 + 10 ** (-g_l * (rw - rl) / 400))
            d2_w = 1.0 / ((GLICKO_Q**2) * (g_l**2) * e_w * (1.0 - e_w))
            rw_new = rw + GLICKO_Q / (1.0 / rdw**2 + 1.0 / d2_w) * g_l * (1.0 - e_w)
            rdw_new = math.sqrt(1.0 / (1.0 / rdw**2 + 1.0 / d2_w))
            rdw_new = max(min(rdw_new, GLICKO_RD_MAX), GLICKO_RD_MIN)

            # Step 2: 败方更新（对胜方）
            g_w = _glicko_g(rdw)
            e_l = 1.0 / (1.0 + 10 ** (-g_w * (rl - rw) / 400))
            d2_l = 1.0 / ((GLICKO_Q**2) * (g_w**2) * e_l * (1.0 - e_l))
            rl_new = rl + GLICKO_Q / (1.0 / rdl**2 + 1.0 / d2_l) * g_w * (0.0 - e_l)
            rdl_new = math.sqrt(1.0 / (1.0 / rdl**2 + 1.0 / d2_l))
            rdl_new = max(min(rdl_new, GLICKO_RD_MAX), GLICKO_RD_MIN)

            r[w_team], r[l_team] = rw_new, rl_new
            rd[w_team], rd[l_team] = rdw_new, rdl_new

        glicko_df = pd.DataFrame(
            [{"TeamID": t, "elo": r[t], "Season": season} for t in r]
        )
        glickos.append(glicko_df)
    return pd.concat(glickos, ignore_index=True)


def _compute_glm_quality(
    regular_data: pd.DataFrame,
    seeds_T1: pd.DataFrame,
    seeds_T2: pd.DataFrame,
    seasons: list,
) -> pd.DataFrame:
    """GLM quality：按 Season 和 men_women 分别拟合，仅对 tournament 球队及曾战胜 tournament 球队的队伍。"""
    regular_data = regular_data.copy()
    regular_data["ST1"] = regular_data.apply(lambda r: f"{int(r['Season'])}/{int(r['T1_TeamID'])}", axis=1)
    regular_data["ST2"] = regular_data.apply(lambda r: f"{int(r['Season'])}/{int(r['T2_TeamID'])}", axis=1)
    seeds_T1 = seeds_T1.copy()
    seeds_T2 = seeds_T2.copy()
    seeds_T1["ST1"] = seeds_T1.apply(lambda r: f"{int(r['Season'])}/{int(r['T1_TeamID'])}", axis=1)
    seeds_T2["ST2"] = seeds_T2.apply(lambda r: f"{int(r['Season'])}/{int(r['T2_TeamID'])}", axis=1)

    st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
    st = st | set(
        regular_data.loc[
            (regular_data["T1_Score"] > regular_data["T2_Score"]) & (regular_data["ST2"].isin(st)),
            "ST1",
        ]
    )

    dt = regular_data.loc[regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)].copy()
    dt["T1_TeamID"] = dt["T1_TeamID"].astype(str)
    dt["T2_TeamID"] = dt["T2_TeamID"].astype(str)
    dt.loc[~dt["ST1"].isin(st), "T1_TeamID"] = "0000"
    dt.loc[~dt["ST2"].isin(st), "T2_TeamID"] = "0000"

    glm_quality = []
    for s in tqdm(seasons, desc="GLM quality"):
        if s >= 2010:
            sub = dt.loc[(dt["Season"] == s) & (dt["men_women"] == 0)]
            if len(sub) > 0:
                try:
                    glm = sm.GLM.from_formula(
                        "PointDiff~-1+T1_TeamID+T2_TeamID",
                        data=sub,
                        family=sm.families.Gaussian(),
                    ).fit()
                    q = pd.DataFrame(glm.params).reset_index()
                    q.columns = ["TeamID", "quality"]
                    q["Season"] = s
                    q = q.loc[q["TeamID"].astype(str).str.contains("T1_", na=False)].copy()
                    q["TeamID"] = q["TeamID"].apply(
                        lambda x: int(re.search(r"(\d{4})", str(x)).group(1)) if re.search(r"(\d{4})", str(x)) else None
                    )
                    q = q.dropna(subset=["TeamID"])
                    glm_quality.append(q)
                except Exception:
                    pass
        if s >= 2003:
            sub = dt.loc[(dt["Season"] == s) & (dt["men_women"] == 1)]
            if len(sub) > 0:
                try:
                    glm = sm.GLM.from_formula(
                        "PointDiff~-1+T1_TeamID+T2_TeamID",
                        data=sub,
                        family=sm.families.Gaussian(),
                    ).fit()
                    q = pd.DataFrame(glm.params).reset_index()
                    q.columns = ["TeamID", "quality"]
                    q["Season"] = s
                    q = q.loc[q["TeamID"].astype(str).str.contains("T1_", na=False)].copy()
                    q["TeamID"] = q["TeamID"].apply(
                        lambda x: int(re.search(r"(\d{4})", str(x)).group(1)) if re.search(r"(\d{4})", str(x)) else None
                    )
                    q = q.dropna(subset=["TeamID"])
                    glm_quality.append(q)
                except Exception:
                    pass

    if not glm_quality:
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])
    return pd.concat(glm_quality, ignore_index=True)


def run_raddar(
    data_dir: str = "march-machine-learning-mania-2026",
    submission_path: str | None = None,
    output_path: str = "submission_raddar.csv",
) -> pd.DataFrame:
    """
    运行完整 Raddar 流水线，生成 submission。

    Args:
        data_dir: 数据目录
        submission_path: SampleSubmissionStage2.csv 路径，默认 data_dir 下
        output_path: 输出文件路径

    Returns:
        含 ID, Pred 的 DataFrame
    """
    data_dir = Path(data_dir)
    if submission_path is None:
        submission_path = data_dir / "SampleSubmissionStage2.csv"

    # ------ 1. 加载数据 ------
    regular_results, tourney_results, seeds = _load_data(str(data_dir))
    regular_data = prepare_data(regular_results)
    tourney_data = prepare_data(tourney_results)
    tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]

    # ------ 2. Seeds ------
    seeds_T1 = seeds.copy()
    seeds_T2 = seeds.copy()
    seeds_T1["T1_seed"] = seeds_T1["Seed"].apply(extract_seed_number)
    seeds_T2["T2_seed"] = seeds_T2["Seed"].apply(extract_seed_number)
    seeds_T1 = seeds_T1[["Season", "TeamID", "T1_seed"]].rename(columns={"TeamID": "T1_TeamID"})
    seeds_T2 = seeds_T2[["Season", "TeamID", "T2_seed"]].rename(columns={"TeamID": "T2_TeamID"})
    tourney_data = tourney_data.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

    # ------ 3. Box 特征 ss_T1, ss_T2 ------
    ss = regular_data.groupby(["Season", "T1_TeamID"])[BOXCOLS].mean().reset_index()
    ss_T1 = ss.copy()
    ss_T1.columns = ["Season", "T1_TeamID"] + [
        "T1_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in BOXCOLS
    ]
    ss_T2 = ss.copy()
    ss_T2.columns = ["Season", "T2_TeamID"] + [
        "T2_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in BOXCOLS
    ]
    tourney_data = tourney_data.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")

    # ------ 4. Elo ------
    elos = _compute_elo(regular_data, seeds)
    elos_T1 = elos.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
    elos_T2 = elos.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})
    tourney_data = tourney_data.merge(elos_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(elos_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

    # ------ 5. GLM quality ------
    seasons = sorted(seeds["Season"].unique())
    glm_quality = _compute_glm_quality(regular_data, seeds_T1, seeds_T2, seasons)
    glm_quality_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
    glm_quality_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
    tourney_data = tourney_data.merge(glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(glm_quality_T2, on=["Season", "T2_TeamID"], how="left")

    # ------ 6. 特征列 ------
    feat_avail = [f for f in FEATURES if f in tourney_data.columns]

    # ------ 7. Leave-one-season-out XGBoost ------
    oof_preds, oof_targets, oof_seasons = [], [], []
    for val_season in tqdm(seasons, desc="XGB LOO"):
        if val_season not in tourney_data["Season"].values:
            continue
        tr = tourney_data.loc[tourney_data["Season"] != val_season]
        val = tourney_data.loc[tourney_data["Season"] == val_season]
        if tr.empty or val.empty:
            continue
        X_tr = tr[feat_avail].fillna(0)
        y_tr = tr["PointDiff"]
        X_val = val[feat_avail].fillna(0)
        y_val = val["PointDiff"]
        dm_tr = xgb.DMatrix(X_tr, label=y_tr)
        dm_val = xgb.DMatrix(X_val)
        model = xgb.train(
            XGB_PARAMS,
            dm_tr,
            num_boost_round=NUM_ROUNDS,
            verbose_eval=False,
        )
        preds = model.predict(dm_val)
        oof_preds.extend(preds.tolist())
        oof_targets.extend(y_val.tolist())
        oof_seasons.extend([val_season] * len(preds))

    # ------ 8. Spline 校准 ------
    dat = sorted(zip(oof_preds, [t > 0 for t in oof_targets]), key=lambda x: x[0])
    pred_clip, label = list(zip(*dat))
    spline_model = UnivariateSpline(np.clip(pred_clip, -SPLINE_T, SPLINE_T), label, k=5)
    spline_fit = np.clip(spline_model(np.clip(oof_preds, -SPLINE_T, SPLINE_T)), 0.01, 0.99)

    # Brier 诊断（与 vilnius 一致）：竞赛核心评分标准，用于评估 spline 校准质量
    oof_labels = np.array(oof_targets) > 0
    print(f"OOF Brier: {brier_score_loss(oof_labels, spline_fit):.8f}")
    for s in sorted(set(oof_seasons)):
        mask = np.array(oof_seasons) == s
        print(s, np.round(brier_score_loss(oof_labels[mask], spline_fit[mask]), 5))

    # ------ 9. 构造预测 X (SampleSubmissionStage2) ------
    sub = pd.read_csv(submission_path)
    X = sub.copy()
    X["Season"] = X["ID"].apply(lambda s: int(s.split("_")[0]))
    X["T1_TeamID"] = X["ID"].apply(lambda s: int(s.split("_")[1]))
    X["T2_TeamID"] = X["ID"].apply(lambda s: int(s.split("_")[2]))
    # men_women 与 vilnius 预测阶段一致：T1 首位'1'(男1xxx)→0，否则(女3xxx)→1
    X["men_women"] = X["T1_TeamID"].apply(lambda t: 0 if str(t)[0] == "1" else 1)
    X = X.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
    X = X.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")
    X = X.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
    X = X.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
    X["Seed_diff"] = X["T2_seed"] - X["T1_seed"]
    X = X.merge(glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
    X = X.merge(glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
    X = X.merge(elos_T1, on=["Season", "T1_TeamID"], how="left")
    X = X.merge(elos_T2, on=["Season", "T2_TeamID"], how="left")
    X["elo_diff"] = X["T1_elo"] - X["T2_elo"]

    # ------ 10. 预测 ------
    X_feat = X[feat_avail].fillna(0)
    dm_pred = xgb.DMatrix(X_feat)
    model_full = xgb.train(
        XGB_PARAMS,
        xgb.DMatrix(tourney_data[feat_avail].fillna(0), label=tourney_data["PointDiff"]),
        num_boost_round=NUM_ROUNDS,
        verbose_eval=False,
    )
    pred_margin = model_full.predict(dm_pred)
    pred_prob = np.clip(spline_model(np.clip(pred_margin, -SPLINE_T, SPLINE_T)), 0.01, 0.99)

    out = pd.DataFrame({"ID": sub["ID"], "Pred": pred_prob})
    out.to_csv(output_path, index=False)
    return out


def run_historical_comparison(
    data_dir: str = "march-machine-learning-mania-2026",
) -> dict:
    """
    历史测试：对比「LOSO OOF」与「单模型（仅用过去赛季）」两种方式的 Brier 分数。

    - LOSO: 每赛季用「排除该赛季」的模型预测，spline 校准后算 Brier
    - 单模型: 每赛季用「仅过去赛季」训练的模型预测（无未来泄露），spline 校准后算 Brier

    Returns:
        {"loso_brier": float, "single_brier": float, "loso_per_season": dict, "single_per_season": dict}
    """
    data_dir = Path(data_dir)
    # 复用 run_raddar 的数据准备逻辑，但只做到特征为止
    regular_results, tourney_results, seeds = _load_data(str(data_dir))
    regular_data = prepare_data(regular_results)
    tourney_data = prepare_data(tourney_results)
    tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]

    seeds_T1 = seeds.copy()
    seeds_T2 = seeds.copy()
    seeds_T1["T1_seed"] = seeds_T1["Seed"].apply(extract_seed_number)
    seeds_T2["T2_seed"] = seeds_T2["Seed"].apply(extract_seed_number)
    seeds_T1 = seeds_T1[["Season", "TeamID", "T1_seed"]].rename(columns={"TeamID": "T1_TeamID"})
    seeds_T2 = seeds_T2[["Season", "TeamID", "T2_seed"]].rename(columns={"TeamID": "T2_TeamID"})
    tourney_data = tourney_data.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

    ss = regular_data.groupby(["Season", "T1_TeamID"])[BOXCOLS].mean().reset_index()
    ss_T1 = ss.copy()
    ss_T1.columns = ["Season", "T1_TeamID"] + [
        "T1_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in BOXCOLS
    ]
    ss_T2 = ss.copy()
    ss_T2.columns = ["Season", "T2_TeamID"] + [
        "T2_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in BOXCOLS
    ]
    tourney_data = tourney_data.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")

    elos = _compute_elo(regular_data, seeds)
    elos_T1 = elos.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
    elos_T2 = elos.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})
    tourney_data = tourney_data.merge(elos_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(elos_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

    seasons = sorted(seeds["Season"].unique())
    glm_quality = _compute_glm_quality(regular_data, seeds_T1, seeds_T2, seasons)
    glm_quality_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
    glm_quality_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
    tourney_data = tourney_data.merge(glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(glm_quality_T2, on=["Season", "T2_TeamID"], how="left")

    feat_avail = [f for f in FEATURES if f in tourney_data.columns]

    # ---------- 1. LOSO OOF Brier ----------
    oof_preds, oof_targets, oof_seasons = [], [], []
    for val_season in tqdm(seasons, desc="LOSO"):
        if val_season not in tourney_data["Season"].values:
            continue
        tr = tourney_data.loc[tourney_data["Season"] != val_season]
        val = tourney_data.loc[tourney_data["Season"] == val_season]
        if tr.empty or val.empty:
            continue
        dm_tr = xgb.DMatrix(tr[feat_avail].fillna(0), label=tr["PointDiff"])
        dm_val = xgb.DMatrix(val[feat_avail].fillna(0))
        model = xgb.train(XGB_PARAMS, dm_tr, num_boost_round=NUM_ROUNDS, verbose_eval=False)
        preds = model.predict(dm_val)
        oof_preds.extend(preds.tolist())
        oof_targets.extend(val["PointDiff"].tolist())
        oof_seasons.extend([val_season] * len(preds))

    dat = sorted(zip(oof_preds, [t > 0 for t in oof_targets]), key=lambda x: x[0])
    pred_clip, label = list(zip(*dat))
    spline_loso = UnivariateSpline(np.clip(pred_clip, -SPLINE_T, SPLINE_T), label, k=5)
    spline_fit_loso = np.clip(spline_loso(np.clip(oof_preds, -SPLINE_T, SPLINE_T)), 0.01, 0.99)
    oof_labels = np.array(oof_targets) > 0
    loso_brier = brier_score_loss(oof_labels, spline_fit_loso)
    loso_per_season = {}
    for s in sorted(set(oof_seasons)):
        mask = np.array(oof_seasons) == s
        loso_per_season[s] = float(brier_score_loss(oof_labels[mask], spline_fit_loso[mask]))

    # ---------- 2. 单模型（仅过去赛季）Brier ----------
    single_preds, single_targets, single_seasons = [], [], []
    for val_season in tqdm(seasons, desc="单模型(过去赛季)"):
        past = [x for x in seasons if x < val_season]
        if not past:
            continue
        tr = tourney_data.loc[tourney_data["Season"].isin(past)]
        val = tourney_data.loc[tourney_data["Season"] == val_season]
        if tr.empty or val.empty:
            continue
        dm_tr = xgb.DMatrix(tr[feat_avail].fillna(0), label=tr["PointDiff"])
        dm_val = xgb.DMatrix(val[feat_avail].fillna(0))
        model = xgb.train(XGB_PARAMS, dm_tr, num_boost_round=NUM_ROUNDS, verbose_eval=False)
        preds = model.predict(dm_val)
        single_preds.extend(preds.tolist())
        single_targets.extend(val["PointDiff"].tolist())
        single_seasons.extend([val_season] * len(preds))

    dat_s = sorted(zip(single_preds, [t > 0 for t in single_targets]), key=lambda x: x[0])
    pred_clip_s, label_s = list(zip(*dat_s))
    spline_single = UnivariateSpline(np.clip(pred_clip_s, -SPLINE_T, SPLINE_T), label_s, k=5)
    spline_fit_single = np.clip(spline_single(np.clip(single_preds, -SPLINE_T, SPLINE_T)), 0.01, 0.99)
    single_labels = np.array(single_targets) > 0
    single_brier = brier_score_loss(single_labels, spline_fit_single)
    single_per_season = {}
    for s in sorted(set(single_seasons)):
        mask = np.array(single_seasons) == s
        single_per_season[s] = float(brier_score_loss(single_labels[mask], spline_fit_single[mask]))

    # ---------- 打印对比 ----------
    print("\n========== 历史测试：LOSO vs 单模型(过去赛季) ==========")
    print(f"LOSO OOF Brier (整体):     {loso_brier:.8f}")
    print(f"单模型(过去赛季) Brier:   {single_brier:.8f}")
    print("\n按赛季对比 (LOSO / 单模型):")
    for s in sorted(set(loso_per_season) | set(single_per_season)):
        lo = loso_per_season.get(s, float("nan"))
        si = single_per_season.get(s, float("nan"))
        if np.isnan(lo):
            print(f"  {s}:   -     / {si:.5f}")
        elif np.isnan(si):
            print(f"  {s}: {lo:.5f} /   -   ")
        else:
            print(f"  {s}: {lo:.5f} / {si:.5f}  (Δ={si-lo:+.5f})")
    print("=" * 50)

    return {
        "loso_brier": loso_brier,
        "single_brier": single_brier,
        "loso_per_season": loso_per_season,
        "single_per_season": single_per_season,
    }


def _apply_pred_boost(probs: np.ndarray, boost: float) -> np.ndarray:
    """对称置信度提升：向更近的极端（0 或 1）推进 boost 比例的距离。"""
    result = np.array(probs, dtype=float)
    mask_high = result > 0.5
    mask_low = result < 0.5
    result[mask_high] = result[mask_high] + boost * (1.0 - result[mask_high])
    result[mask_low] = result[mask_low] - boost * result[mask_low]
    return np.clip(result, 0.01, 0.99)


def run_realistic_backtest(
    data_dir: str = "march-machine-learning-mania-2026",
    test_years: list[int] | None = None,
    use_glm: bool = True,
    rating_system: str = "elo",
    pred_boost: float | None = None,
) -> dict[int, float]:
    """
    真实模拟历史回测：每年独立运行，模拟「若在该年参赛，只用此前数据」的 Brier 分数。

    对每个 test_year (默认 2021–2025)：
    1. 训练集 = Season < test_year（严格只用过去）
    2. 在训练集内做 LOSO，得到 OOF 预测，拟合 spline 校准曲线（无未来泄露）
    3. 用全量训练集训练最终模型，预测 test_year
    4. 用步骤 2 的 spline 校准后，（可选）应用 pred_boost，再计算 Brier

    use_glm: 若 False，不使用 GLM quality 特征（用于消融实验）
    rating_system: "elo" 或 "glicko"，队伍强度评分方式
    pred_boost: 预测置信度提升比例，如 0.10(10%)、0.05(5%)；对称向 0/1 推进；None 则不提升

    Returns:
        {test_year: brier_score, ...}
    """
    if test_years is None:
        test_years = [2021, 2022, 2023, 2024, 2025]

    data_dir = Path(data_dir)
    regular_results, tourney_results, seeds = _load_data(str(data_dir))
    regular_data = prepare_data(regular_results)
    tourney_data = prepare_data(tourney_results)
    tourney_data = tourney_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]]

    seeds_T1 = seeds.copy()
    seeds_T2 = seeds.copy()
    seeds_T1["T1_seed"] = seeds_T1["Seed"].apply(extract_seed_number)
    seeds_T2["T2_seed"] = seeds_T2["Seed"].apply(extract_seed_number)
    seeds_T1 = seeds_T1[["Season", "TeamID", "T1_seed"]].rename(columns={"TeamID": "T1_TeamID"})
    seeds_T2 = seeds_T2[["Season", "TeamID", "T2_seed"]].rename(columns={"TeamID": "T2_TeamID"})
    tourney_data = tourney_data.merge(seeds_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(seeds_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

    ss = regular_data.groupby(["Season", "T1_TeamID"])[BOXCOLS].mean().reset_index()
    ss_T1 = ss.copy()
    ss_T1.columns = ["Season", "T1_TeamID"] + [
        "T1_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in BOXCOLS
    ]
    ss_T2 = ss.copy()
    ss_T2.columns = ["Season", "T2_TeamID"] + [
        "T2_avg_" + c.replace("T1_", "").replace("T2_", "opponent_") for c in BOXCOLS
    ]
    tourney_data = tourney_data.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")

    if rating_system.lower() == "glicko":
        ratings_df = _compute_glicko(regular_data, seeds)
    else:
        ratings_df = _compute_elo(regular_data, seeds)
    ratings_T1 = ratings_df.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
    ratings_T2 = ratings_df.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})
    tourney_data = tourney_data.merge(ratings_T1, on=["Season", "T1_TeamID"], how="left")
    tourney_data = tourney_data.merge(ratings_T2, on=["Season", "T2_TeamID"], how="left")
    tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

    seasons_all = sorted(seeds["Season"].unique())
    if use_glm:
        glm_quality = _compute_glm_quality(regular_data, seeds_T1, seeds_T2, seasons_all)
        glm_quality_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
        glm_quality_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
        tourney_data = tourney_data.merge(glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
        tourney_data = tourney_data.merge(glm_quality_T2, on=["Season", "T2_TeamID"], how="left")

    feat_avail = [f for f in FEATURES if f in tourney_data.columns]

    results: dict[int, float] = {}
    for test_year in tqdm(test_years, desc="真实回测"):
        past = [s for s in seasons_all if s < test_year]
        if not past:
            continue
        tr = tourney_data.loc[tourney_data["Season"].isin(past)]
        val = tourney_data.loc[tourney_data["Season"] == test_year]
        if tr.empty or val.empty:
            continue

        # 1. 在训练集内 LOSO，得到 OOF，拟合 spline（仅用过去数据，无泄露）
        oof_p, oof_t = [], []
        for leave_out in past:
            tr_loo = tr.loc[tr["Season"] != leave_out]
            val_loo = tr.loc[tr["Season"] == leave_out]
            if tr_loo.empty or val_loo.empty:
                continue
            dm_tr = xgb.DMatrix(
                tr_loo[feat_avail].fillna(0), label=tr_loo["PointDiff"]
            )
            dm_val = xgb.DMatrix(val_loo[feat_avail].fillna(0))
            m = xgb.train(XGB_PARAMS, dm_tr, num_boost_round=NUM_ROUNDS, verbose_eval=False)
            oof_p.extend(m.predict(dm_val).tolist())
            oof_t.extend(val_loo["PointDiff"].tolist())

        if len(oof_p) < 10:
            continue
        dat = sorted(zip(oof_p, [t > 0 for t in oof_t]), key=lambda x: x[0])
        pred_clip, lbl = list(zip(*dat))
        spline = UnivariateSpline(np.clip(pred_clip, -SPLINE_T, SPLINE_T), lbl, k=5)

        # 2. 全量训练集训练，预测 test_year
        dm_tr_full = xgb.DMatrix(
            tr[feat_avail].fillna(0), label=tr["PointDiff"]
        )
        dm_val = xgb.DMatrix(val[feat_avail].fillna(0))
        model = xgb.train(XGB_PARAMS, dm_tr_full, num_boost_round=NUM_ROUNDS, verbose_eval=False)
        preds = model.predict(dm_val)

        # 3. 校准 + 可选 pred_boost + Brier
        probs = np.clip(spline(np.clip(preds, -SPLINE_T, SPLINE_T)), 0.01, 0.99)
        if pred_boost is not None and pred_boost > 0:
            probs = _apply_pred_boost(probs, pred_boost)
        labels = (val["PointDiff"].values > 0).astype(float)
        b = float(brier_score_loss(labels, probs))
        results[test_year] = b

    # 打印结果
    parts = []
    if not use_glm:
        parts.append("无GLM")
    if rating_system.lower() == "glicko":
        parts.append("Glicko")
    if pred_boost is not None and pred_boost > 0:
        parts.append(f"pred_boost={int(pred_boost*100)}%")
    tag = " + " + " + ".join(parts) if parts else ""
    print(f"\n========== 真实模拟历史回测 (5 次独立运行){tag} ==========")
    print("每年：用该年之前的数据训练+校准，在该年上测试 Brier")
    for y in sorted(results):
        print(f"  {y}: 训练 2003..{y-1} → 测试 {y}  Brier = {results[y]:.5f}")
    if results:
        avg_b = sum(results.values()) / len(results)
        print(f"  近{len(results)}年平均 Brier: {avg_b:.5f}")
    print("=" * 50)

    return results
