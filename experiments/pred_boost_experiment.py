"""
Pred 置信度提升实验（10% / 5% 对称增大）

在 spline 校准后的 Pred 上，向更近的极端（0 或 1）推进一定比例，观察 Brier 变化。
- pred > 0.5: 向 1 推进
- pred < 0.5: 向 0 推进

Baseline 不跑，使用预先存储结果。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.raddar_pipeline import run_realistic_backtest

DATA_DIR = ROOT / "march-machine-learning-mania-2026"
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]

# Baseline（无 boost，用户已记录）
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
    print("Pred 置信度提升实验（10% / 5% 对称）")
    print("=" * 60)

    # ----- 5% boost -----
    print("\n>>> 5% 对称 boost")
    r5 = run_realistic_backtest(
        data_dir=str(DATA_DIR),
        test_years=TEST_YEARS,
        pred_boost=0.05,
    )
    avg5 = sum(r5.values()) / len(r5) if r5 else float("nan")

    # ----- 10% boost -----
    print("\n>>> 10% 对称 boost")
    r10 = run_realistic_backtest(
        data_dir=str(DATA_DIR),
        test_years=TEST_YEARS,
        pred_boost=0.10,
    )
    avg10 = sum(r10.values()) / len(r10) if r10 else float("nan")

    # ----- 汇总对比 -----
    print("\n" + "=" * 60)
    print("汇总对比 (Brier，越低越好)")
    print("=" * 60)
    print(f"{'年份':<8} {'Baseline':<14} {'5% boost':<14} {'10% boost':<14}")
    print("-" * 55)
    for y in TEST_YEARS:
        b = BASELINE.get(y, float("nan"))
        e5 = r5.get(y, float("nan"))
        e10 = r10.get(y, float("nan"))
        print(f"  {y}     {b:.5f}        {e5:.5f}        {e10:.5f}")
    print("-" * 55)
    print(f"  平均     {BASELINE_AVG:.5f}        {avg5:.5f}        {avg10:.5f}")
    print()
    print("解读：若 boost 降低平均 Brier，说明对称增大有效")
    print("=" * 60)


if __name__ == "__main__":
    main()
