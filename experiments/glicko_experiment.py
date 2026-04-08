"""
Glicko vs Elo 评分系统实验

测试流程与之前一致：真实模拟历史回测（2021–2025，每年用该年之前数据训练+校准，在该年上测试 Brier）。

实验配置：
  - Baseline (Elo): 有 GLM + Elo  →  近5年平均 Brier 0.16548（已记录）
  - Glicko: 有 GLM + Glicko 替代 Elo  →  对比 Brier 变化

若 Glicko 优于 Elo，说明引入 RD（评分可靠性）的 Glicko 能更好刻画队伍强度。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.raddar_pipeline import run_realistic_backtest

DATA_DIR = ROOT / "march-machine-learning-mania-2026"
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# Baseline 结果（Elo + GLM，用户已记录）
BASELINE_ELO = {
    2021: 0.17776,
    2022: 0.18640,
    2023: 0.18737,
    2024: 0.15569,
    2025: 0.12020,
}
BASELINE_ELO_AVG = 0.16548


def main():
    print("=" * 60)
    print("Glicko vs Elo 实验")
    print("=" * 60)

    # ----- Elo（Baseline 复跑，确保一致）-----
    print("\n>>> Elo + GLM (Baseline)")
    r_elo = run_realistic_backtest(
        data_dir=str(DATA_DIR),
        test_years=TEST_YEARS,
        rating_system="elo",
    )
    avg_elo = sum(r_elo.values()) / len(r_elo) if r_elo else float("nan")

    # ----- Glicko -----
    print("\n>>> Glicko + GLM")
    r_glicko = run_realistic_backtest(
        data_dir=str(DATA_DIR),
        test_years=TEST_YEARS,
        rating_system="glicko",
    )
    avg_glicko = sum(r_glicko.values()) / len(r_glicko) if r_glicko else float("nan")

    # ----- 汇总对比 -----
    print("\n" + "=" * 60)
    print("汇总对比 (Brier，越低越好)")
    print("=" * 60)
    print(f"{'年份':<8} {'Elo(Baseline)':<18} {'Glicko':<16} {'Δ':<10}")
    print("-" * 55)
    for y in TEST_YEARS:
        e = r_elo.get(y, float("nan"))
        g = r_glicko.get(y, float("nan"))
        delta = g - e if not (e != e or g != g) else float("nan")
        delta_str = f"{delta:+.5f}" if delta == delta else "-"
        print(f"  {y}     {e:.5f}           {g:.5f}       {delta_str}")
    print("-" * 55)
    print(f"  平均     {avg_elo:.5f}           {avg_glicko:.5f}       {avg_glicko - avg_elo:+.5f}")
    print()
    print("解读：Δ < 0 表示 Glicko 优于 Elo")
    print("=" * 60)


if __name__ == "__main__":
    main()
