"""
GLM 消融实验（仅无 GLM vs 有 GLM）

目的：验证 GLM quality 特征的贡献。

实验配置：
  - Baseline (已记录): 有 GLM + Elo  →  近5年平均 Brier 0.16548
  - 无 GLM: Elo only  →  看去掉 GLM 后 Brier 变化
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.raddar_pipeline import run_realistic_backtest

DATA_DIR = ROOT / "march-machine-learning-mania-2026"
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

BASELINE = {
    2021: 0.17776,
    2022: 0.18640,
    2023: 0.18737,
    2024: 0.15569,
    2025: 0.12020,
}
BASELINE_AVG = 0.16548


def main():
    print("=" * 60)
    print("GLM 消融实验")
    print("=" * 60)

    print("\n>>> 无 GLM（Elo only）")
    r_no_glm = run_realistic_backtest(
        data_dir=str(DATA_DIR),
        test_years=TEST_YEARS,
        use_glm=False,
    )
    avg_no_glm = sum(r_no_glm.values()) / len(r_no_glm) if r_no_glm else float("nan")

    print("\n" + "=" * 60)
    print("汇总对比 (Brier)")
    print("=" * 60)
    print(f"{'年份':<8} {'Baseline(有GLM)':<18} {'无GLM':<16}")
    print("-" * 45)
    for y in TEST_YEARS:
        b = BASELINE.get(y, float("nan"))
        e = r_no_glm.get(y, float("nan"))
        print(f"  {y}     {b:.5f}              {e:.5f}")
    print("-" * 45)
    print(f"  平均     {BASELINE_AVG:.5f}              {avg_no_glm:.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
