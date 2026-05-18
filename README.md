# 基于机器学习的 Lorenz 混沌系统状态预测与预测边界分析

**English Title:** Machine Learning for Lorenz Chaotic System State Prediction and Forecasting Boundary Analysis

## 项目背景

本项目是《人工智能导论》期末小组项目，主题属于数学建模与 AI4SCI 方向。Lorenz 系统是非线性动力系统和混沌理论中的经典模型，具有确定性方程、非线性耦合和初值敏感性等特点。

项目原始任务是：使用 Python 数值模拟生成 Lorenz 系统数据，并将连续动力系统预测问题转化为监督学习回归任务，即给定当前状态变量 `x_t`, `y_t`, `z_t`，预测短期未来的 `x_next = x(t + 10Δt)`。

增强后的项目进一步扩展为完整三维状态转移预测：

```text
(x_t, y_t, z_t) -> (x_{t+k}, y_{t+k}, z_{t+k})
```

重点不再只是证明短期预测可以取得很高精度，而是分析机器学习模型在 Lorenz 混沌系统中的预测边界：短期局部状态转移可学习，但预测步长增大、递归多步预测和初始条件变化会暴露长期预测限制。

## Lorenz 方程

Lorenz 系统由以下三个常微分方程组成：

$$
\frac{dx}{dt}=\sigma(y-x)
$$

$$
\frac{dy}{dt}=x(\rho-z)-y
$$

$$
\frac{dz}{dt}=xy-\beta z
$$

本项目采用经典参数：

$$
\sigma=10,\quad \rho=28,\quad \beta=\frac{8}{3}
$$

## 数据生成方法

数据由 Lorenz 方程数值模拟生成：

1. 使用 `scipy.integrate.solve_ivp` 求解 Lorenz 微分方程；
2. 原始训练轨迹初始状态为 `(1, 1, 1)`；
3. 模拟时间范围为 `[0, 50]`；
4. 采样点数为 `10000`；
5. 构造监督学习数据集。

原始任务使用单变量目标：

- 输入特征：`x_t`, `y_t`, `z_t`
- 预测目标：`x_next = x(t + 10Δt)`

增强任务使用完整状态目标：

- 输入特征：`x_t`, `y_t`, `z_t`
- 预测目标：`x_{t+k}`, `y_{t+k}`, `z_{t+k}`

## 模型角色

本项目保留并增强以下模型，各自承担不同分析角色：

1. **Persistence Model**  
   连续性基线，直接使用当前状态作为未来状态预测，用于衡量系统短期连续性本身能带来多少预测能力。

2. **Linear Regression**  
   局部线性化基线，用线性映射近似局部状态转移。

3. **Random Forest Regressor**  
   传统非线性机器学习强基线，用于比较树模型在不同预测步长下的稳定性。

4. **MLPRegressor**  
   基本神经网络状态转移模型，用多层感知机学习非线性状态转移映射。

5. **Residual MLP**  
   预测状态变化量 `Δs = s_{t+k} - s_t`，再还原未来状态，贴近数值积分中的“当前状态 + 增量”思想。

6. **Echo State Network / Reservoir Computing**  
   Dynamical time-series model，用固定随机 reservoir 提取时序动态特征，再训练线性 readout。

7. **SINDy / Sparse Identification of Nonlinear Dynamics**  
   从轨迹数据中估计导数，在多项式候选函数库中用稀疏回归恢复 Lorenz 方程结构，用于回答模型是否学到了动力学规律。

8. **Hybrid Physical-ML Correction**  
   构造参数有偏的不完美 Lorenz 物理模型，再用机器学习学习真实系统与物理预测之间的残差，体现 AI4SCI 中“物理模型 + 数据驱动误差修正”的思路。

项目没有盲目添加 KNN、SVR、XGBoost 等普通模型，因为本项目重点是预测任务设计、动力系统状态转移结构和误差分析，而不是堆叠模型列表。

## 新增实验

### 1. Prediction horizon sensitivity

测试 horizon：

```text
k = [1, 5, 10, 20, 50, 100, 200, 500]
```

对每个 `k` 构造完整三维状态预测任务：

```text
(x_t, y_t, z_t) -> (x_{t+k}, y_{t+k}, z_{t+k})
```

输出文件：

- `results/horizon_results.csv`
- `figures/horizon_vs_rmse.png`

### 2. 三维状态预测

增强实验将模型输出从单一 `x` 值扩展为完整状态向量 `(x, y, z)`，并计算：

```text
RMSE_state = sqrt(mean(||s_true - s_pred||^2))
```

同时保存每个状态分量的 RMSE、MAE、R²。

### 3. Recursive multi-step rollout

训练 one-step 状态转移模型：

```text
(x_t, y_t, z_t) -> (x_{t+1}, y_{t+1}, z_{t+1})
```

然后从测试段某个真实状态出发，只给模型一次真实初始状态，后续每一步都使用模型自己的预测作为下一步输入，连续 rollout 500 步。

输出文件：

- `results/rollout_results.csv`
- `figures/rollout_error.png`
- `figures/rollout_x_comparison.png`
- `figures/rollout_attractor.png`

### 4. Initial-condition generalization

模型仍然主要在 `(1, 1, 1)` 轨迹上训练，不对新初始条件重新训练。新增测试轨迹包括：

```text
(1.01, 1, 1)
(1.1, 1, 1)
(-5, 5, 20)
(10, 10, 10)
```

输出文件：

- `results/initial_condition_generalization.csv`
- `figures/initial_condition_rmse.png`

### 5. Model enhancement experiments

模型增强实验进一步测试什么样的结构更适合学习 Lorenz 系统的连续状态转移映射：

- 对输出 `Y=(x,y,z)` 或 `ΔY` 使用 `StandardScaler`；
- 比较 direct MLP 与 residual MLP；
- 比较 `[32,32]`、`[64,64]`、`[128,128]`、`[64,64,64]` 等 MLP 容量；
- 比较 ReLU 与 tanh 激活函数；
- 加入轻量 Echo State Network 作为动力系统时间序列模型。

输出文件：

- `results/model_enhancement_results.csv`
- `results/enhanced_rollout_results.csv`
- `figures/model_enhancement_horizon_rmse.png`
- `figures/enhanced_rollout_error.png`
- `figures/enhanced_rollout_x_comparison.png`

### 6. SINDy sparse dynamics identification

SINDy 实验将项目从“预测未来状态”进一步扩展到“从数据中识别动力学结构”。实验使用中心差分估计 `dx/dt, dy/dt, dz/dt`，构造候选库：

```text
1, x, y, z, x^2, xy, xz, y^2, yz, z^2
```

再使用 sequential thresholded least squares 恢复 Lorenz 方程的稀疏非零项。

输出文件：

- `results/sindy_coefficients.csv`
- `results/sindy_equation_error.csv`
- `figures/sindy_coefficients.png`
- `figures/sindy_rollout_x_comparison.png`

### 7. Hybrid physical-ML correction

Hybrid 实验模拟真实科学建模中“已有物理模型但参数有偏”的情况。真实系统使用 `rho=28`，不完美物理模型故意使用 `rho=26`。机器学习模型不直接替代物理模型，而是学习：

```text
residual = s_true(t+1) - s_physics(t+1)
hybrid_prediction = s_physics(t+1) + ML_residual
```

比较对象包括 imperfect physics、pure ML residual MLP、Hybrid RF 和 Hybrid MLP。评价指标新增 valid prediction time，并用 Lorenz63 最大 Lyapunov 指数约 `0.9` 将物理时间换算为 Lyapunov time。

输出文件：

- `results/hybrid_metrics.csv`
- `results/hybrid_rollout_results.csv`
- `results/valid_prediction_time.csv`
- `results/long_term_statistics.csv`
- `figures/hybrid_one_step_metrics.png`
- `figures/hybrid_rollout_error.png`
- `figures/valid_prediction_time.png`
- `figures/long_term_distribution_comparison.png`

## 主要结果

原始单变量短期预测任务中，非线性模型在 `x(t+10Δt)` 上取得了很高精度：

| 模型 | RMSE | MAE | R² |
|---|---:|---:|---:|
| Persistence Model | 2.1653 | 1.7657 | 0.9208 |
| Linear Regression | 0.5679 | 0.4556 | 0.9946 |
| Random Forest | 0.0887 | 0.0504 | 0.9999 |
| MLP | 0.0779 | 0.0574 | 0.9999 |

增强实验表明，完整三维状态预测在短 horizon 下仍较准确，但随着 horizon 增大误差明显上升。例如：

| Horizon | 模型 | State RMSE | Mean R² |
|---:|---|---:|---:|
| 10 | Random Forest | 0.2987 | 0.9996 |
| 10 | MLP | 0.4572 | 0.9991 |
| 100 | Random Forest | 4.3280 | 0.9164 |
| 100 | MLP | 2.2369 | 0.9781 |
| 500 | Random Forest | 12.7868 | 0.2607 |
| 500 | MLP | 13.6858 | 0.1591 |

Recursive rollout 也显示，一步预测模型在递归使用时会出现明显误差累积。500 步 rollout 结束时，Linear Regression、Random Forest、MLP 的累计 state RMSE 分别约为 `20.2040`、`14.3255`、`18.0711`。

模型增强实验进一步表明，Residual MLP 和输出标准化能显著改善中短期三维状态预测。例如 horizon=10 时，`MLP-residual-relu-64x64x64` 的 state RMSE 为 `0.0772`；horizon=100 时，该模型 state RMSE 为 `1.4133`，优于原始 MLP 的 `2.2369`。在 enhanced rollout 中，`MLP-residual-tanh-64x64` 的 500 步累计 state RMSE 约为 `10.7975`，优于原始 MLP rollout 的 `18.0711`。ESN 作为 reservoir computing 模型在本实验中表现中等，rollout 累计 state RMSE 约为 `14.7144`，可作为动力系统时间序列模型的补充对照。

SINDy 实验从轨迹数据中恢复出接近真实 Lorenz 方程的稀疏结构：

```text
dx/dt ≈ -9.9943x + 9.9943y
dy/dt ≈ 27.9517x - 0.9906y - 0.9986xz
dz/dt ≈ 0.9992xy - 2.6646z
```

Hybrid correction 实验进一步表明，机器学习可以作为不完美物理模型的残差修正器。真实系统使用 `rho=28`，不完美物理模型使用 `rho=26` 时，一步 state RMSE 从 `0.0786` 降至 `0.0003`（Hybrid RF）和 `0.0006`（Hybrid MLP）。在 500 步 rollout 中，imperfect physics 的有效预测时间约为 `0.35` 个 Lyapunov time，pure ML residual MLP 约为 `1.86` 个 Lyapunov time，而 Hybrid RF 和 Hybrid MLP 在 500 步窗口内均未超过失效阈值，对应至少 `2.25` 个 Lyapunov time。

因此，本项目的主要结论是：机器学习模型可以较好学习 Lorenz 系统的短期局部状态转移映射；SINDy 可以从数据中识别动力学方程结构；Hybrid physical-ML correction 可以显著改善参数有偏物理模型的短中期推演。但预测步长增大、递归多步预测和初始条件变化仍会使预测难度上升，不能将短期高精度夸大为“机器学习已经解决混沌系统长期预测”。

## 如何运行

建议在已安装依赖后，从项目根目录运行 notebook：

```bash
jupyter notebook lorenz_chaos_prediction.ipynb
```

或使用 nbconvert 从头执行：

```bash
jupyter nbconvert --to notebook --execute lorenz_chaos_prediction.ipynb --inplace
```

## 图表说明

图表保存在 `figures/` 文件夹。

原始图表包括：

- `lorenz_attractor.png`：Lorenz 三维吸引子图；
- `lorenz_time_series.png`：状态变量时间序列图；
- `correlation_heatmap.png`：相关性热力图；
- `prediction_scatter_linear.png`：线性回归真实值 vs 预测值散点图；
- `prediction_scatter_rf.png`：随机森林真实值 vs 预测值散点图；
- `time_series_prediction.png`：测试集局部时间序列预测对比图；
- `residuals.png`：随机森林残差图；
- `model_metrics_comparison.png`：模型评价指标对比图；
- `feature_importance.png`：随机森林特征重要性图。

增强图表包括：

- `horizon_vs_rmse.png`：prediction horizon 与 state RMSE 的关系；
- `rollout_error.png`：recursive rollout 误差累积曲线；
- `rollout_x_comparison.png`：rollout 中真实 `x(t)` 与预测 `x(t)` 对比；
- `rollout_attractor.png`：真实轨迹与 rollout 预测轨迹三维对比；
- `initial_condition_rmse.png`：不同初始条件下的泛化误差对比；
- `model_enhancement_horizon_rmse.png`：Residual MLP、输出标准化和 ESN 在不同 horizon 下的对比；
- `enhanced_rollout_error.png`：增强模型 recursive rollout 误差曲线；
- `enhanced_rollout_x_comparison.png`：增强模型 rollout 中 `x(t)` 的真实值与预测值对比。
- `sindy_coefficients.png`：SINDy 恢复的 Lorenz 方程系数热力图；
- `sindy_rollout_x_comparison.png`：SINDy 恢复方程 rollout 中 `x(t)` 的真实值与预测值对比；
- `hybrid_one_step_metrics.png`：imperfect physics、pure ML 与 hybrid correction 的一步预测误差对比；
- `hybrid_rollout_error.png`：hybrid correction 在 recursive rollout 中的累计误差曲线；
- `valid_prediction_time.png`：不同模型以 Lyapunov time 归一化的有效预测时间；
- `long_term_distribution_comparison.png`：真实轨迹与 rollout 轨迹的长期分布对比。

## 结果文件

结果表格保存在 `results/` 文件夹。

原始结果包括：

- `lorenz_supervised_dataset.csv`
- `missing_values.csv`
- `descriptive_stats.csv`
- `standardization_check.csv`
- `model_metrics.csv`
- `feature_importance.csv`
- `summary.json`

增强结果包括：

- `horizon_results.csv`
- `rollout_results.csv`
- `initial_condition_generalization.csv`
- `model_enhancement_results.csv`
- `enhanced_rollout_results.csv`
- `sindy_coefficients.csv`
- `sindy_equation_error.csv`
- `hybrid_metrics.csv`
- `hybrid_rollout_results.csv`
- `valid_prediction_time.csv`
- `long_term_statistics.csv`

## 项目文件结构

```text
lorenz-ml-project/
├── README.md
├── requirements.txt
├── lorenz_chaos_prediction.ipynb
├── report.tex
├── report.pdf
├── figures/
│   ├── lorenz_attractor.png
│   ├── lorenz_time_series.png
│   ├── correlation_heatmap.png
│   ├── prediction_scatter_linear.png
│   ├── prediction_scatter_rf.png
│   ├── time_series_prediction.png
│   ├── residuals.png
│   ├── model_metrics_comparison.png
│   ├── feature_importance.png
│   ├── horizon_vs_rmse.png
│   ├── rollout_error.png
│   ├── rollout_x_comparison.png
│   ├── rollout_attractor.png
│   ├── initial_condition_rmse.png
│   ├── model_enhancement_horizon_rmse.png
│   ├── enhanced_rollout_error.png
│   ├── enhanced_rollout_x_comparison.png
│   ├── sindy_coefficients.png
│   ├── sindy_rollout_x_comparison.png
│   ├── hybrid_one_step_metrics.png
│   ├── hybrid_rollout_error.png
│   ├── valid_prediction_time.png
│   └── long_term_distribution_comparison.png
└── results/
    ├── lorenz_supervised_dataset.csv
    ├── missing_values.csv
    ├── descriptive_stats.csv
    ├── standardization_check.csv
    ├── model_metrics.csv
    ├── feature_importance.csv
    ├── summary.json
    ├── horizon_results.csv
    ├── rollout_results.csv
    ├── initial_condition_generalization.csv
    ├── model_enhancement_results.csv
    ├── enhanced_rollout_results.csv
    ├── sindy_coefficients.csv
    ├── sindy_equation_error.csv
    ├── hybrid_metrics.csv
    ├── hybrid_rollout_results.csv
    ├── valid_prediction_time.csv
    └── long_term_statistics.csv
```

## 小组信息

- 小组成员：钟兴涛、黄宇轩、郝致祺
- 课程名称：人工智能导论
