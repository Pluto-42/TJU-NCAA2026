# NCAA2026 可复用模块

基于 Raddar (vilnius-ncaa)、goto_conversion、手动覆盖策略整理的模块，便于 2026 赛季直接复用。

## 模块概览

| 模块 | 功能 |
|------|------|
| `optimal_strategy` | set_optimalStrategy, apply_manual_overrides, get_roundOfMatch 等 |
| `raddar_utils` | prepare_data, extract_seed_number |
| `goto_utils` | load_probability_tables |

## 使用示例

```python
from src import prepare_data, set_optimalStrategy, apply_manual_overrides

# Raddar 数据准备
regular_data = prepare_data(regular_results)

# 风险队押注
submission_df = set_optimalStrategy(
    submission_df, mens_seeds, womens_seeds,
    risk_teams=[1179], risk_team_to_win_rounds=[2]
)

# 固定对阵覆盖
submission_df = apply_manual_overrides(submission_df, {
    "2025_1124_1280": 0.9818,
    "2025_1140_1433": 0.9803,
})
```
