# 参考 Notebook 索引

本文件夹存放 NCAA 2026 竞赛相关的参考 notebook，供开工方案中的实现参考。

| 文件 | 用途 | 备注 |
|------|------|------|
| **vilnius-ncaa.ipynb** | Raddar 主流程 | Elo + GLM quality + XGBoost + spline，src/raddar_utils.py 来源 |
| **ncaa2025-7th-solution.ipynb** | Raddar + set_optimalStrategy | 主体为 Raddar，末尾叠加 33.3% 策略覆盖 |
| **ncaa2025-3th-solution.ipynb** | 538 submission + set_optimalStrategy | 直接读 538data submission.csv 后覆盖 |
| **ncaa2025-1st-solution.ipynb** | 第一名方案 | 复杂 pipeline |
| **goto-conversion-winning-solution.ipynb** | goto_conversion 核心算法 | P(win\|reach) 公式 + preprocessed_goto_conversion，历史验证用 |
| **updated-goto-conversion-winning-solution.ipynb** | 538 submission + set_optimalStrategy | 读 538 预填 submission，不做概率表→pairwise 计算 |
| **ncaa2026-public-baseline-v2.ipynb** | 2026 官方 baseline | 参考 baseline 结构 |
| **hoops-i-did-it-again.ipynb** | AutoGluon + 赛季统计 | 仅 Kaggle 数据，无 goto_conversion |
