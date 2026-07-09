# 基于 MATLAB 的高维混沌系统长期预测建模

- **课程名称：** MATLAB及应用（强）
- **组长：** 钟兴涛（25120617）
- **组员：** 唐亦明（25120638）、戴云天（25120636）、任宇航（25120699）、黄宇轩（25120619）

## 项目简介

本项目基于 MATLAB 完成 Lorenz 混沌系统长期预测实验。实验先使用 Lorenz-63 系统验证短期监督学习和递归推演之间的差异，再扩展到 10 维 Lorenz-96 系统，比较 Baseline LSTM、PINN LSTM、Transformer、Hybrid LSTM 与 Ultimate Hybrid 在 500-step recursive rollout 中的稳定性。

核心结论是：one-step prediction 精度不能直接代表长期预测能力。在高维混沌系统中，将物理演化模型作为基础预测，再让神经网络学习残差，并结合物理约束和多步训练，可以提升长期 rollout 稳定性。

## MATLAB 运行环境

- MATLAB R2026a 或兼容版本；
- Deep Learning Toolbox；
- Statistics and Machine Learning Toolbox；
- 支持中文的 XeLaTeX 环境。

## 一键运行

在仓库根目录运行完整实验：

```bash
matlab -batch "run_all_matlab"
```

快速检查主要代码路径：

```bash
matlab -batch "run_all_matlab('smoke')"
```

生成基于 Word 模板封面的最终论文：

```bash
matlab -batch "build_final_pdf"
```

## 输出目录

- `results/*.csv`：Lorenz-63 指标、rollout、horizon 与 hybrid 结果；
- `results/advanced/*.csv`：Lorenz-96 高级模型汇总与各模型 rollout 结果；
- `figures/*.png`：论文图像；
- `cover_filled.docx`：填写后的 Word 封面；
- `cover.pdf`：由 Word 模板第一页导出的封面 PDF；
- `report_body.pdf`：LaTeX 编译得到的正文 PDF；
- `report.pdf`：封面与正文合并后的最终论文 PDF。

## 文件结构

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
│   ├── tests/
│   │   └── verifyFinalSubmission.m
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
    ├── lstm_phase_summary.csv
    ├── lstm_phase1_rollout.csv
    ├── lstm_phase2_rollout.csv
    ├── lstm_phase3_summary.csv
    ├── lstm_phase3_w10_rollout.csv
    ├── lstm_phase3_w20_rollout.csv
    ├── lstm_phase3_w50_rollout.csv
    └── advanced/
        ├── summary.csv
        ├── Baseline_LSTM_rollout.csv
        ├── PINN_LSTM_rollout.csv
        ├── Refined_SS_LSTM_rollout.csv
        ├── Transformer_rollout.csv
        ├── Hybrid_LSTM_rollout.csv
        └── Ultimate_Hybrid_rollout.csv
```

## 主要结论

### Lorenz-63：短期可学但 rollout 会累积误差

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
| Baseline LSTM | 15.26 |
| PINN LSTM | 477.29 |
| Refined SS LSTM | 15.16 |
| Transformer | 23.25 |
| Hybrid LSTM | 78.42 |
| **Ultimate Hybrid** | **14.42** |

Ultimate Hybrid 在高级架构对比中取得最低最终累计 RMSE。结果说明，物理基础预测、残差学习、Transformer 序列建模、PINN 约束和 refined scheduled sampling 的组合能更好地抑制长期 rollout 发散。
