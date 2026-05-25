# 基于监督学习与物理残差修正的 Lorenz 混沌系统预测研究

**English Title:** Lorenz Chaotic System Prediction with Supervised Learning and Physics Residual Correction

## 项目简介

本项目以 Lorenz 混沌系统为对象，使用 AI 导论课程中的回归、树模型和神经网络方法进行监督学习预测，并进一步通过 Hybrid correction 学习不完美物理模型的残差，展示机器学习在科学建模中的误差修正作用。

项目主线是：

1. 使用 Lorenz 方程生成混沌系统轨迹；
2. 构造监督学习任务：用当前状态预测未来状态；
3. 使用 Linear Regression、Random Forest 和 MLP 进行直接预测；
4. 分析 prediction horizon 和 recursive rollout 中的误差累积；
5. 使用 Residual MLP 学习状态增量；
6. 使用 Hybrid RF / Hybrid MLP 学习不完美物理模型的残差。

核心结论：短期局部状态转移可学，但长期逐点预测受混沌误差放大限制；在当前 `rho=26` 的参数偏差设定下，Hybrid correction 能显著改善不完美物理模型的预测效果。



## 数据来源

数据由 Lorenz 方程数值模拟生成，不使用外部数据集：

```math
\frac{dx}{dt}=\sigma(y-x)
```

```math
\frac{dy}{dt}=x(\rho-z)-y
```

```math
\frac{dz}{dt}=xy-\beta z
```

真实系统参数为：

```math
\sigma=10,\quad \rho=28,\quad \beta=\frac{8}{3}
```

基础轨迹设置：

- 初始状态：`(1, 1, 1)`；
- 模拟时间：`[0, 50]`；
- 采样点数：`10000`；
- 数值求解：`scipy.integrate.solve_ivp`；
- 训练/测试切分：按时间顺序前 80% / 后 20%。

## 主要方法

### 1. 直接状态预测

输入当前状态：

```text
s_t = (x_t, y_t, z_t)
```

预测未来状态：

```text
s_{t+k} = (x_{t+k}, y_{t+k}, z_{t+k})
```

使用模型：

- Linear Regression；
- Random Forest；
- MLP。

### 2. Residual MLP

Residual MLP 不直接预测未来状态，而是预测状态变化量：

```text
Δs = s_{t+k} - s_t
s_hat_{t+k} = s_t + MLP(s_t)
```



### 3. Hybrid correction

真实系统使用 `rho=28`，不完美物理模型使用 `rho=26`。先由不完美物理模型预测：

```text
s_phys(t+1)
```

再训练机器学习模型学习残差：

```text
residual = s_true(t+1) - s_phys(t+1)
hybrid_prediction = s_phys(t+1) + ML_residual
```

比较对象：

- Imperfect physics；
- Pure ML Residual MLP；
- Hybrid RF；
- Hybrid MLP。

## 实验结构

1. **直接监督学习预测**  
   比较 Linear Regression、Random Forest、MLP 的短期预测能力。

2. **Prediction horizon 分析**  
   代表性 horizon：`k=10, 100, 500`。完整结果见 `results/horizon_results.csv`。

3. **Recursive rollout 分析**  
   只给定初始真实状态，后续不断使用模型自身预测作为下一步输入，观察误差累积。

4. **Residual MLP 对比**  
   比较 Direct MLP 与 Residual MLP，分析状态增量学习是否改善中短期预测。

5. **不完美物理模型与 Hybrid correction**  
   比较单独使用参数有偏物理模型、纯机器学习模型和物理残差修正模型。

## 主要结果

### 直接短期预测

原始课程任务 `x(t+10Δt)` 的结果来自 `results/model_metrics.csv`：

| 模型 | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.5457 | 0.4437 | 0.9952 |
| Random Forest | 0.0805 | 0.0507 | 0.9999 |
| MLP | 0.0757 | 0.0599 | 0.9999 |

这说明短期局部映射可以被非线性模型拟合，但不能推出长期混沌预测已经解决。

### Prediction horizon

代表性三维状态预测结果来自 `results/horizon_results.csv`：

| Horizon | 模型 | State RMSE | Mean R² |
|---:|---|---:|---:|
| 10 | Linear Regression | 4.7791 | 0.9028 |
| 10 | Random Forest | 0.3425 | 0.9995 |
| 10 | MLP | 0.4130 | 0.9993 |
| 100 | Linear Regression | 15.4505 | -0.0793 |
| 100 | Random Forest | 3.1292 | 0.9569 |
| 100 | MLP | 2.1529 | 0.9800 |
| 500 | Linear Regression | 15.2982 | -0.0292 |
| 500 | Random Forest | 10.9789 | 0.4714 |
| 500 | MLP | 14.0980 | 0.1245 |

随着 horizon 增大，各模型误差明显上升，说明普通监督学习预测存在时间尺度边界。

### Recursive rollout

500 步 rollout 最终累计 State RMSE 来自 `results/rollout_results.csv`：

| 模型 | Rollout 步数 | 最终累计 State RMSE |
|---|---:|---:|
| Linear Regression | 500 | 20.2244 |
| Random Forest | 500 | 14.5356 |
| MLP | 500 | 19.8142 |

Rollout 中模型不断使用自己的预测作为输入，早期误差会进入后续输入并持续累积。

### Residual MLP

Direct MLP 与 Residual MLP 对比来自 `results/model_enhancement_results.csv`：

| Horizon | 模型 | State RMSE | Mean R² |
|---:|---|---:|---:|
| 10 | Direct MLP | 0.4130 | 0.9993 |
| 10 | Residual MLP | 0.0833 | 1.0000 |
| 100 | Direct MLP | 2.1529 | 0.9800 |
| 100 | Residual MLP | 0.8306 | 0.9970 |
| 500 | Direct MLP | 14.0980 | 0.1245 |
| 500 | Residual MLP | 12.0584 | 0.3609 |

Residual MLP 在中短期预测中更好，但在很长 horizon 下仍会退化。

### Hybrid correction

一步预测结果来自 `results/hybrid_metrics.csv`：

| 模型 | State RMSE | Mean R² |
|---|---:|---:|
| Imperfect physics | 0.0786 | 0.9999746657 |
| Pure ML Residual MLP | 0.0102 | 0.9999995570 |
| Hybrid RF | 0.0003 | 0.9999999996 |
| Hybrid MLP | 0.0006 | 0.9999999986 |

500 步 rollout 有效预测时间来自 `results/valid_prediction_time.csv`：

| 模型 | 有效步数 | 物理时间 | Lyapunov time | 最终累计 State RMSE |
|---|---:|---:|---:|---:|
| Imperfect physics | 77 | 0.3850 | 0.3465 | 19.6795 |
| Pure ML Residual MLP | 413 | 2.0652 | 1.8587 | 11.1571 |
| Hybrid RF | 500 | 2.5003 | 2.2502 | 0.7774 |
| Hybrid MLP | 500 | 2.5003 | 2.2502 | 0.0862 |

在当前参数偏差 `rho=26` 的实验设置下，Hybrid correction 显著降低一步误差，并在 500 步 rollout 中保持更长有效预测时间。

## 如何运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行 notebook：

```bash
jupyter notebook lorenz_chaos_prediction.ipynb
```

或从头执行 notebook：

```bash
jupyter nbconvert --to notebook --execute lorenz_chaos_prediction.ipynb --inplace
```

从 CSV 重新生成报告表格核对片段：

```bash
python scripts/generate_report_tables.py
```

输出文件：

```text
results/generated_report_tables.tex
```

编译报告：

```bash
xelatex report.tex
```

## 项目文件结构

```text
lorenz-ml-project/
├── README.md
├── requirements.txt
├── lorenz_chaos_prediction.ipynb
├── report.tex
├── report.pdf
├── scripts/
│   └── generate_report_tables.py
├── figures/
│   ├── lorenz_attractor.png
│   ├── horizon_vs_rmse.png
│   ├── rollout_error.png
│   ├── model_enhancement_horizon_rmse.png
│   ├── hybrid_one_step_metrics.png
│   ├── hybrid_rollout_error.png
│   └── valid_prediction_time.png
└── results/
    ├── model_metrics.csv
    ├── horizon_results.csv
    ├── rollout_results.csv
    ├── model_enhancement_results.csv
    ├── hybrid_metrics.csv
    ├── valid_prediction_time.csv
    ├── summary.json
    └── generated_report_tables.tex
```

## 结论

本项目把 Lorenz 混沌系统预测重构为一个清晰的 AI 导论 + AI for Science 小项目：先用 Linear Regression、Random Forest 和 MLP 做直接监督学习预测，分析普通黑箱模型在 horizon 和 rollout 上的预测边界；再用 Residual MLP 和 Hybrid correction 学习残差，突出机器学习修正不完美物理模型的科学建模价值。

## 小组信息

- 小组成员：钟兴涛、黄宇轩、郝致祺
- 课程名称：人工智能导论
