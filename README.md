# 基于 LLM 辅助与机器学习的 Lorenz 混沌系统短期状态预测研究

**English Title:** LLM-Assisted Machine Learning for Short-Term State Prediction of the Lorenz Chaotic System

## 项目背景

本项目是《人工智能导论》期末小组项目，主题属于数学建模与 AI4SCI 方向。Lorenz 系统是非线性动力系统和混沌理论中的经典模型，具有确定性方程、非线性耦合和初值敏感性等特点。项目目标是使用 Python 数值模拟生成 Lorenz 系统数据，并用机器学习模型预测系统短期未来状态。

本项目将连续动力系统预测问题转化为监督学习回归任务：给定当前时刻状态变量 `x_t`, `y_t`, `z_t`，预测短期未来的 `x_next = x(t + 10Δt)`。

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

数据不是从外部下载，而是由 Lorenz 方程数值模拟生成：

1. 使用 `scipy.integrate.solve_ivp` 求解 Lorenz 微分方程；
2. 初始状态设置为 `(1, 1, 1)`；
3. 模拟时间范围为 `[0, 50]`；
4. 采样点数为 `10000`；
5. 构造监督学习数据集：
   - 输入特征：`x_t`, `y_t`, `z_t`
   - 预测目标：`x_next = x(t + 10Δt)`

最终数据集包含 `9990` 条样本，训练集 `7992` 条，测试集 `1998` 条。

## 模型路线

本项目比较了四种预测方法：

1. **Persistence Model**  
   朴素时间序列基线，直接使用当前 `x_t` 预测未来 `x_next`。

2. **Linear Regression**  
   简单 Baseline Model，用线性组合预测未来状态。

3. **Random Forest Regressor**  
   主模型，能够捕捉非线性关系，训练稳定且可解释性较好。

4. **MLPRegressor**  
   扩展模型，使用简单多层感知机神经网络进行非线性回归。

## 主要结果

| 模型 | RMSE | MAE | R² |
|---|---:|---:|---:|
| Persistence Model | 2.1653 | 1.7657 | 0.9208 |
| Linear Regression | 0.5679 | 0.4556 | 0.9946 |
| Random Forest | 0.0887 | 0.0504 | 0.9999 |
| MLP | 0.0779 | 0.0574 | 0.9999 |

结果表明，非线性模型 Random Forest 和 MLP 明显优于线性回归 Baseline，说明机器学习模型能够较好地学习 Lorenz 系统的短期局部演化规律。

## 图表说明

图表保存在 `figures/` 文件夹：

- `lorenz_attractor.png`：Lorenz 三维吸引子图；
- `lorenz_time_series.png`：状态变量时间序列图；
- `correlation_heatmap.png`：相关性热力图；
- `prediction_scatter_linear.png`：线性回归真实值 vs 预测值散点图；
- `prediction_scatter_rf.png`：随机森林真实值 vs 预测值散点图；
- `time_series_prediction.png`：测试集局部时间序列预测对比图；
- `residuals.png`：随机森林残差图；
- `model_metrics_comparison.png`：模型评价指标对比图；
- `feature_importance.png`：随机森林特征重要性图。

结果表格保存在 `results/` 文件夹。

## 如何运行 notebook

建议先创建虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

启动 Jupyter Notebook：

```bash
jupyter notebook
```

打开并从头运行：

```text
lorenz_chaos_prediction.ipynb
```

运行完成后会自动生成或更新：

- `figures/` 中的所有图表；
- `results/` 中的数据表、指标表和摘要文件。

## 如何编译报告

本项目报告源文件为：

```text
report.tex
```

需要使用 XeLaTeX 编译中文 LaTeX：

```bash
xelatex report.tex
xelatex report.tex
```

编译后生成：

```text
report.pdf
```

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
│   └── feature_importance.png
└── results/
    ├── lorenz_supervised_dataset.csv
    ├── missing_values.csv
    ├── descriptive_stats.csv
    ├── standardization_check.csv
    ├── model_metrics.csv
    ├── feature_importance.csv
    └── summary.json
```

## 小组信息

- 小组成员：钟兴涛、黄宇轩、郝致祺
- 课程名称：人工智能导论
