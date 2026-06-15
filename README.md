# 基于大语言模型辅助的高维混沌系统长期预测建模

- **副标题：** 以 Lorenz-63 与 Lorenz-96 系统为例
- **课程名称：** 人工智能导论
- **小组成员：** 钟兴涛、黄宇轩、郝致祺

## 项目简介

本项目研究混沌动力系统的长期多步预测问题。项目先用 Lorenz-63 系统说明普通监督学习模型可以拟合短期局部状态转移，但在 recursive rollout 中会因为误差累积而逐渐偏离真实轨迹；随后将重点扩展到 10 维 Lorenz-96 系统，比较 Baseline LSTM、PINN LSTM、Transformer、Hybrid LSTM 与 Ultimate Hybrid 在 500-step rollout 中的长期稳定性。

核心结论是：**one-step prediction 精度不能代表长期预测能力**。在高维混沌系统中，更有效的路线不是只依赖黑箱神经网络，而是以物理模型作为演化基底，让深度学习模型学习残差，并结合物理约束和多步训练提升 rollout 稳定性。

## 研究主线

1. 使用 Lorenz-63 验证“短期可学、长期受限”的基础现象；
2. 构造 Lorenz-96 (10D) 高维混沌系统数据集；
3. 使用历史窗口预测下一时刻状态，并以 500-step 累计 RMSE 评价长期 rollout；
4. 从 Baseline LSTM 逐步加入残差预测、LayerNorm、gradient clipping 和 scheduled sampling；
5. 比较 PINN 物理约束、Transformer 序列建模、Hybrid physics 残差修正等高级架构；
6. 提出 Ultimate Hybrid：不完美 RK4 物理求解器 + Transformer 残差 + PINN 损失 + refined scheduled sampling。

## 数据与任务定义

### Lorenz-63 基础实验

Lorenz-63 系统由三个耦合常微分方程生成：

```math
\frac{dx}{dt}=\sigma(y-x)
```

```math
\frac{dy}{dt}=x(\rho-z)-y
```

```math
\frac{dz}{dt}=xy-\beta z
```

基础参数为：

```math
\sigma=10,\quad \rho=28,\quad \beta=\frac{8}{3}
```

该部分用于说明 Linear Regression、Random Forest、MLP 等普通监督学习模型在短期预测中表现较好，但 prediction horizon 增大或进行 recursive rollout 后误差会明显累积。

### Lorenz-96 高维实验

Lorenz-96 使用 10 维状态变量：

```math
\frac{dx_i}{dt}=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F,\quad i=1,\dots,N
```

其中下标按周期边界处理，本文设置：

```text
N = 10
F_true = 8.0
```

模型输入为长度为 `w` 的历史窗口，输出为下一时刻完整 10 维状态或状态增量。主要评价指标是 500-step recursive rollout 的最终累计 RMSE。

## 主要方法

### 1. Lorenz-63 监督学习基线

- Linear Regression：课程基础回归模型；
- Random Forest：树模型非线性基线；
- MLP：神经网络直接预测；
- Residual MLP 与 Hybrid correction：保留在基础 notebook 和结果文件中，用于说明残差学习思想。

### 2. Lorenz-96 LSTM 阶段优化

- Baseline LSTM：直接预测下一时刻绝对状态；
- Residual LSTM：预测状态增量 `delta x`，再与当前状态相加；
- LayerNorm 与 gradient clipping：提升训练稳定性；
- Scheduled Sampling：缓解训练时使用真实历史、rollout 时使用模型预测造成的分布偏移；
- Window ablation：比较不同历史窗口长度对长期推演稳定性的影响。

### 3. 高级架构对比

- PINN LSTM：将 Lorenz-96 物理方程作为损失约束；
- Refined SS LSTM：使用更平滑的 scheduled sampling 策略；
- Transformer：用 self-attention 捕捉历史窗口内部关系；
- Hybrid LSTM：先由不完美物理求解器预测，再由 LSTM 学习残差；
- Ultimate Hybrid：使用 `F_imp=8.5` 的不完美 RK4 求解器提供基础演化，Transformer 学习残差，并结合 PINN 损失与 5 步 refined scheduled sampling。

## 主要结果

### Lorenz-63：短期可学但 rollout 发散

原始短期任务 `x(t+10dt)` 的结果来自 `results/model_metrics.csv`：

| 模型 | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.5457 | 0.4437 | 0.9952 |
| Random Forest | 0.0805 | 0.0507 | 0.9999 |
| MLP | 0.0757 | 0.0599 | 0.9999 |

500 步 recursive rollout 的最终累计 State RMSE 来自 `results/rollout_results.csv`：

| 模型 | 最终累计 State RMSE |
|---|---:|
| Linear Regression | 20.2244 |
| Random Forest | 14.5356 |
| MLP | 19.8142 |

这些结果说明，短期拟合效果好并不意味着模型能够长期稳定推演混沌轨迹。

### Lorenz-96：LSTM 阶段性优化

结果来自论文表格与 `results/lstm_phase*_rollout.csv`：

| 模型阶段 | One-step RMSE | One-step R² | 500-step 最终累计 RMSE |
|---|---:|---:|---:|
| Baseline LSTM（绝对状态预测） | 0.7360 | 0.9531 | 13.9662 |
| Phase 1：残差预测 + LayerNorm + 裁剪 | 0.0417 | 0.9999 | 13.1749 |
| Phase 2：多步损失 + Scheduled Sampling | 0.0356 | 0.9999 | 14.1880 |

Phase 2 的 one-step RMSE 更低，但 500-step rollout 反而更差，进一步说明长期稳定性必须单独评价。

### Lorenz-96：历史窗口长度消融

结果来自 `results/lstm_phase3_summary.csv`：

| 窗口长度 `w` | 500-step 最终累计 RMSE |
|---:|---:|
| 10 | 15.7545 |
| 20 | 16.5752 |
| 50 | 111.0108 |

过长历史窗口 `w=50` 显著恶化长期推演，说明高维混沌序列中的历史信息并非越多越好。

### Lorenz-96：高级架构总体对比

结果来自 `results/advanced/summary.csv`：

| 模型架构 | 500-step 最终累计 RMSE |
|---|---:|
| Baseline LSTM | 148.79 |
| PINN LSTM | 20.18 |
| Refined SS LSTM | 16.79 |
| Transformer | 16.34 |
| Hybrid LSTM | 15.84 |
| **Ultimate Hybrid** | **12.80** |

Ultimate Hybrid 在所有高级架构中取得最低最终累计 RMSE，说明物理基底、残差学习、Transformer 序列建模、PINN 约束和 refined scheduled sampling 的组合能更好地抑制长期 rollout 发散。

## 如何运行

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含 PyTorch：

```text
torch>=2.0,<3
```

运行 Lorenz-63 基础 notebook：

```bash
jupyter notebook lorenz_chaos_prediction.ipynb
```

或从头执行 notebook：

```bash
jupyter nbconvert --to notebook --execute lorenz_chaos_prediction.ipynb --inplace
```

运行 Lorenz-96 LSTM 阶段实验：

```bash
python lorenz96_lstm.py
python lorenz96_lstm_phase2.py
python lorenz96_lstm_phase3.py
```

运行 Lorenz-96 高级架构对比：

```bash
python lorenz96_advanced_experiments.py
```

查看高级结果汇总：

```bash
type results\advanced\summary.csv
```

重新生成报告表格核对片段：

```bash
python scripts/generate_report_tables.py
```

编译课程论文和汇报 slides：

```bash
xelatex report.tex
xelatex report_slides.tex
```

## 项目文件结构

```text
lorenz-ml-project/
├── README.md
├── requirements.txt
├── report.tex
├── report.pdf
├── report_slides.tex
├── report_slides.pdf
├── presentation_script_5min.md
├── lorenz_chaos_prediction.ipynb
├── lorenz96_lstm.py
├── lorenz96_lstm_phase2.py
├── lorenz96_lstm_phase3.py
├── lorenz96_lstm_optim.py
├── lorenz96_advanced_experiments.py
├── append_lstm.py
├── scratch_code.py
├── scripts/
│   └── generate_report_tables.py
├── figures/
│   ├── lorenz_attractor.png
│   ├── correlation_heatmap.png
│   ├── rollout_error.png
│   ├── prediction_scatter_rf.png
│   ├── lorenz96_window_ablation.png
│   ├── lorenz96_advanced_rollout_comparison.png
│   ├── correlation_heatmap_en.png
│   ├── lorenz_attractor_en.png
│   ├── prediction_scatter_rf_en.png
│   └── rollout_error_en.png
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
    └── advanced/
        ├── summary.csv
        ├── Baseline_LSTM_rollout.csv
        ├── PINN_LSTM_rollout.csv
        ├── Refined_SS_LSTM_rollout.csv
        ├── Transformer_rollout.csv
        ├── Hybrid_LSTM_rollout.csv
        └── Ultimate_Hybrid_rollout.csv
```

## 论文结论

本文从 Lorenz-63 的基础监督学习实验出发，说明短期局部状态转移可以被机器学习拟合，但长期 recursive rollout 会受到混沌误差放大的限制。随后，项目将重点转向 Lorenz-96 (10D) 高维混沌系统，围绕 500-step rollout 稳定性设计并比较多种深度学习架构。

最终结果表明，单步预测精度不能代表长期预测能力。更稳健的 AI for Science 建模路线是：以近似物理模型提供主要演化方向，以深度学习模型学习物理模型残差，并通过物理损失和多步训练减少长期外推中的误差累积。
