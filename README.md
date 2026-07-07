# 基于 MATLAB 的高维混沌系统长期预测建模

- **副标题：** 以 Lorenz-63 与 Lorenz-96 系统为例
- **课程名称：** 人工智能导论
- **组长：** 钟兴涛（25120617）
- **组员：** 唐亦明（25120638）、戴云天（25120636）、任宇航（25120699）、黄宇轩（25120619）

## 项目简介

本项目是原 Lorenz 混沌系统预测课程论文的 MATLAB 实现版。研究主线、实验结论、模型排序和最终判断保持不变；本分支的核心变化是将主要实验入口迁移为 MATLAB / Deep Learning Toolbox / Statistics and Machine Learning Toolbox。

项目先用 Lorenz-63 系统说明普通监督学习模型可以拟合短期局部状态转移，但在 recursive rollout 中会因为误差累积而逐渐偏离真实轨迹；随后将重点扩展到 10 维 Lorenz-96 系统，比较 Baseline LSTM、PINN LSTM、Transformer、Hybrid LSTM 与 Ultimate Hybrid 在 500-step rollout 中的长期稳定性。

核心结论是：**one-step prediction 精度不能代表长期预测能力**。在高维混沌系统中，更有效的路线不是只依赖黑箱神经网络，而是以物理模型作为演化基底，让深度学习模型学习残差，并结合物理约束和多步训练提升 rollout 稳定性。

## MATLAB 运行环境

建议环境：

- MATLAB R2026a 或兼容版本；
- Deep Learning Toolbox；
- Statistics and Machine Learning Toolbox；
- 支持中文的 XeLaTeX 环境（用于编译论文）。

## 如何运行 MATLAB 实验

在仓库根目录运行：

```bash
matlab -batch "run_all_matlab"
```

该入口会生成或更新：

- `results/model_metrics.csv`、`results/rollout_results.csv`、`results/horizon_results.csv` 等：由 MATLAB 实际运行得到的 Lorenz-63 结果；
- `results/matlab_lorenz96_residual_lstm_rollout.csv` 与 `results/matlab_lorenz96_residual_lstm_summary.csv`：由 MATLAB 实际训练得到的 Lorenz-96 残差 LSTM 基线结果；
- `results/lorenz96_matlab_run_status.csv`：说明 Lorenz-96 各结果文件的来源；
- `figures/*.png`：论文引用图像；
- `results/generated_report_tables.tex`：由结果 CSV 生成的表格核对片段。

说明：`results/lstm_phase*.csv` 与 `results/advanced/*.csv` 保留为原论文参考结果，用于保持论文结论和模型排序不变；它们不再被 MATLAB 脚本伪装为重新训练输出。

如只想快速检查 MATLAB 路径和主要输出逻辑，可运行：

```bash
matlab -batch "run_all_matlab('smoke')"
```

## 如何生成论文 PDF

正文仍以 LaTeX 源文件 `report.tex` 为母版。推荐流程：

1. 运行 MATLAB 实验：

   ```bash
   matlab -batch "run_all_matlab"
   ```

2. 编译正文 PDF：

   ```bash
   xelatex report.tex
   xelatex report.tex
   ```

3. 生成并合并课程封面与正文：

   ```bash
   matlab -batch "build_final_pdf"
   ```

最终完整论文 PDF 路径：

```text
report.pdf
```

正文 LaTeX 源文件保留为：

```text
report.tex
```

## 项目文件结构

```text
lorenz-ml-project/
├── README.md
├── run_all_matlab.m
├── build_final_pdf.m
├── report.tex
├── report.pdf
├── matlab_src/
│   ├── runLorenz63Experiments.m
│   ├── runLorenz96Experiments.m
│   ├── generateReportTablesMatlab.m
│   ├── buildFinalPdf.m
│   └── utils/
│       ├── cumulativeRmse.m
│       ├── ensureFolder.m
│       ├── lorenz63Rhs.m
│       ├── lorenz96Rhs.m
│       ├── rk4Lorenz96Step.m
│       └── standardizeTrainOnly.m
├── figures/
│   ├── lorenz_attractor.png
│   ├── correlation_heatmap.png
│   ├── rollout_error.png
│   ├── prediction_scatter_rf.png
│   ├── lorenz96_window_ablation.png
│   └── lorenz96_advanced_rollout_comparison.png
└── results/
    ├── model_metrics.csv
    ├── rollout_results.csv
    ├── horizon_results.csv
    ├── hybrid_metrics.csv
    ├── valid_prediction_time.csv
    ├── lstm_phase1_rollout.csv
    ├── lstm_phase2_rollout.csv
    ├── lstm_phase3_summary.csv
    ├── lstm_phase3_w10_rollout.csv
    ├── lstm_phase3_w20_rollout.csv
    ├── lstm_phase3_w50_rollout.csv
    ├── matlab_lorenz96_residual_lstm_rollout.csv
    ├── matlab_lorenz96_residual_lstm_summary.csv
    ├── lorenz96_matlab_run_status.csv
    └── advanced/
        ├── summary.csv
        ├── Baseline_LSTM_rollout.csv
        ├── PINN_LSTM_rollout.csv
        ├── Refined_SS_LSTM_rollout.csv
        ├── Transformer_rollout.csv
        ├── Hybrid_LSTM_rollout.csv
        └── Ultimate_Hybrid_rollout.csv
```

## 研究主线

1. 使用 Lorenz-63 验证“短期可学、长期受限”的基础现象；
2. 构造 Lorenz-96 (10D) 高维混沌系统数据集；
3. 使用历史窗口预测下一时刻状态，并以 500-step 累计 RMSE 评价长期 rollout；
4. 从 Baseline LSTM 逐步加入残差预测、LayerNorm、gradient clipping 和 scheduled sampling；
5. 比较 PINN 物理约束、Transformer 序列建模、Hybrid physics 残差修正等高级架构；
6. 提出 Ultimate Hybrid：不完美 RK4 物理求解器 + Transformer 残差 + PINN 损失 + refined scheduled sampling。

## 主要结果

### Lorenz-63：短期可学但 rollout 发散

| 模型 | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.5383 | 0.4481 | 0.9953 |
| Random Forest | 0.0913 | 0.0640 | 0.9999 |
| MLP | 0.0138 | 0.0108 | 1.0000 |

500 步 recursive rollout 的最终累计 State RMSE：

| 模型 | 最终累计 State RMSE |
|---|---:|
| Linear Regression | 17.2665 |
| Random Forest | 1.7883 |
| MLP | 6.6833 |

### Lorenz-96：高级架构总体对比

| 模型架构 | 500-step 最终累计 RMSE |
|---|---:|
| Baseline LSTM | 148.79 |
| PINN LSTM | 20.18 |
| Refined SS LSTM | 16.79 |
| Transformer | 16.34 |
| Hybrid LSTM | 15.84 |
| **Ultimate Hybrid** | **12.80** |

Ultimate Hybrid 在所有高级架构中取得最低最终累计 RMSE，说明物理基底、残差学习、Transformer 序列建模、PINN 约束和 refined scheduled sampling 的组合能更好地抑制长期 rollout 发散。

## 参考实现说明

原 Python 脚本和 notebook 保留在仓库中作为实现迁移的参考材料与历史溯源，但本分支的主要运行入口是 MATLAB：

```text
run_all_matlab.m
```
