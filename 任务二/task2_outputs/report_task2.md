# 任务二：病害阶段与气象因子相关性分析（自动报告 v2）

## 数据与方法
- 以 `image` 精确对齐拍摄微气候，并叠加当周/前1/前2周周级暴露特征；
- 单变量：Spearman ρ + 互信息，BH-FDR 校正；
- 多变量：有序Logit（如可拟合）、多项Logit（分组CV）、LightGBM（分组CV）；

## 主要结果
- 阶段分布见 `tables/table_stage_counts.csv`。
- 多项Logit 交叉验证指标见 `multinomial_logit_metrics.json`。
- LightGBM 交叉验证指标见 `lightgbm_metrics.json`，重要性图见 `figures/fig_lightgbm_importance_beautified.png`。

## 图表清单（可直接用于论文）
- figures/fig_stage_distribution.png
- figures/fig_weather_hist.png
- figures/fig_boxplots_by_stage.png
- figures/fig_spearman_heatmap.png
- figures/fig_significant_features.png
- figures/fig_lightgbm_importance_beautified.png
- figures/fig_pdp_top_features.png
