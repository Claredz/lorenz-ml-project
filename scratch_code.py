from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font="SimHei")
plt.rcParams["axes.unicode_minus"] = False
# ---
sigma = 10
rho = 28
beta = 8 / 3
initial_state = [1.0, 1.0, 1.0]
t_start, t_end = 0, 50
n_points = 10000
t_eval = np.linspace(t_start, t_end, n_points)

def lorenz_system(t, state, sigma=sigma, rho=rho, beta=beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

solution = solve_ivp(
    lorenz_system,
    (t_start, t_end),
    initial_state,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-9,
    atol=1e-12,
)

trajectory = pd.DataFrame({
    "time": solution.t,
    "x": solution.y[0],
    "y": solution.y[1],
    "z": solution.y[2],
})

trajectory.head()
# ---
prediction_horizon = 10
supervised = pd.DataFrame({
    "x_t": trajectory["x"].iloc[:-prediction_horizon].to_numpy(),
    "y_t": trajectory["y"].iloc[:-prediction_horizon].to_numpy(),
    "z_t": trajectory["z"].iloc[:-prediction_horizon].to_numpy(),
    "x_next": trajectory["x"].iloc[prediction_horizon:].to_numpy(),
})

supervised.to_csv(RESULTS_DIR / "lorenz_supervised_dataset.csv", index=False)
print(f"数据集维度: {supervised.shape}")
supervised.head()
# ---
missing_values = supervised.isna().sum()
missing_values.to_csv(RESULTS_DIR / "missing_values.csv", header=["missing_count"])
print("各列缺失值数量：")
print(missing_values)

descriptive_stats = supervised.describe().T
selected_stats = descriptive_stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
selected_stats.to_csv(RESULTS_DIR / "descriptive_stats.csv")
selected_stats
# ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(trajectory["x"], trajectory["y"], trajectory["z"], linewidth=0.4)
ax.set_title("Lorenz 系统三维吸引子")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "lorenz_attractor.png", dpi=300, bbox_inches="tight")
plt.show()
# ---
plt.figure(figsize=(10, 5))
plt.plot(trajectory["time"], trajectory["x"], label="x", linewidth=0.8)
plt.plot(trajectory["time"], trajectory["y"], label="y", linewidth=0.8)
plt.plot(trajectory["time"], trajectory["z"], label="z", linewidth=0.8)
plt.xlabel("时间")
plt.ylabel("状态变量")
plt.title("Lorenz 系统状态变量时间序列")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "lorenz_time_series.png", dpi=300, bbox_inches="tight")
plt.show()
# ---
plt.figure(figsize=(7, 6))
corr = supervised.corr()
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", square=True)
plt.title("Lorenz 监督学习数据相关性热力图")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
# ---
feature_cols = ["x_t", "y_t", "z_t"]
target_col = "x_next"

X = supervised[feature_cols].to_numpy()
y = supervised[target_col].to_numpy()

split_index = int(len(supervised) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

standardization_check = pd.DataFrame({
    "feature": feature_cols,
    "train_scaled_mean": X_train_scaled.mean(axis=0),
    "train_scaled_std": X_train_scaled.std(axis=0),
})
standardization_check.to_csv(RESULTS_DIR / "standardization_check.csv", index=False)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
standardization_check
# ---
def evaluate_regression(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

predictions = {}
metrics = {}

persistence_pred = X_test[:, 0]
predictions["Persistence"] = persistence_pred
metrics["Persistence"] = evaluate_regression(y_test, persistence_pred)

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_pred = linear_model.predict(X_test_scaled)
predictions["Linear Regression"] = linear_pred
metrics["Linear Regression"] = evaluate_regression(y_test, linear_pred)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
predictions["Random Forest"] = rf_pred
metrics["Random Forest"] = evaluate_regression(y_test, rf_pred)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=600,
    early_stopping=True,
    random_state=RANDOM_STATE,
)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)
predictions["MLP"] = mlp_pred
metrics["MLP"] = evaluate_regression(y_test, mlp_pred)

metrics_df = pd.DataFrame(metrics).T
metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv")
metrics_df
# ---
def plot_prediction_scatter(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.55)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="理想预测线 y=x")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()

plot_prediction_scatter(y_test, linear_pred, "线性回归：真实值 vs 预测值", "prediction_scatter_linear.png")
plot_prediction_scatter(y_test, rf_pred, "随机森林：真实值 vs 预测值", "prediction_scatter_rf.png")
# ---
n_display = 250
plt.figure(figsize=(11, 5))
plt.plot(y_test[:n_display], label="真实值", linewidth=1.5)
plt.plot(linear_pred[:n_display], label="线性回归预测", linewidth=1.0, alpha=0.8)
plt.plot(rf_pred[:n_display], label="随机森林预测", linewidth=1.0, alpha=0.8)
plt.xlabel("测试集时间步")
plt.ylabel("x(t + 10Δt)")
plt.title("测试集局部时间序列预测对比")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "time_series_prediction.png", dpi=300, bbox_inches="tight")
plt.show()
# ---
residuals = y_test - rf_pred
plt.figure(figsize=(7, 5))
plt.scatter(rf_pred, residuals, s=10, alpha=0.55)
plt.axhline(0, color="red", linestyle="--", linewidth=1.2)
plt.xlabel("随机森林预测值")
plt.ylabel("残差")
plt.title("随机森林残差图")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "residuals.png", dpi=300, bbox_inches="tight")
plt.show()
# ---
metrics_plot = metrics_df.reset_index().rename(columns={"index": "model"})
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
    sns.barplot(data=metrics_plot, x="model", y=metric, ax=ax)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
plt.suptitle("模型评价指标对比")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_metrics_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
# ---
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)
importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

plt.figure(figsize=(6, 4))
sns.barplot(data=importance_df, x="importance", y="feature")
plt.xlabel("特征重要性")
plt.ylabel("输入变量")
plt.title("随机森林特征重要性")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
importance_df
# ---
summary = {
    "sigma": sigma,
    "rho": rho,
    "beta": beta,
    "initial_state": initial_state,
    "t_start": t_start,
    "t_end": t_end,
    "n_points": n_points,
    "prediction_horizon": prediction_horizon,
    "delta_t": float(t_eval[1] - t_eval[0]),
    "effective_prediction_time": float((t_eval[1] - t_eval[0]) * prediction_horizon),
    "dataset_rows": int(len(supervised)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "missing_values": {k: int(v) for k, v in missing_values.to_dict().items()},
    "metrics": metrics,
}

with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

summary
# ---
from sklearn.multioutput import MultiOutputRegressor

STATE_COLS = ["x", "y", "z"]
FEATURE_COLS_STATE = ["x_t", "y_t", "z_t"]
TARGET_COLS_STATE = ["x_future", "y_future", "z_future"]
HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500]
GENERALIZATION_INITIAL_CONDITIONS = [
    (1.01, 1.0, 1.0),
    (1.1, 1.0, 1.0),
    (-5.0, 5.0, 20.0),
    (10.0, 10.0, 10.0),
]


def simulate_lorenz(initial_state, t_start=t_start, t_end=t_end, n_points=n_points, sigma=sigma, rho=rho, beta=beta):
    t_eval_local = np.linspace(t_start, t_end, n_points)
    solution_local = solve_ivp(
        lambda t, state: lorenz_system(t, state, sigma=sigma, rho=rho, beta=beta),
        (t_start, t_end),
        initial_state,
        t_eval=t_eval_local,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    return pd.DataFrame({
        "time": solution_local.t,
        "x": solution_local.y[0],
        "y": solution_local.y[1],
        "z": solution_local.y[2],
    })


def build_state_dataset(trajectory_df, horizon):
    states = trajectory_df[STATE_COLS].to_numpy()
    return pd.DataFrame({
        "x_t": states[:-horizon, 0],
        "y_t": states[:-horizon, 1],
        "z_t": states[:-horizon, 2],
        "x_future": states[horizon:, 0],
        "y_future": states[horizon:, 1],
        "z_future": states[horizon:, 2],
    })


def state_rmse(y_true, y_pred):
    state_error = np.linalg.norm(y_true - y_pred, axis=1)
    return float(np.sqrt(np.mean(state_error ** 2)))


def evaluate_state_prediction(y_true, y_pred):
    errors = y_true - y_pred
    state_abs_error = np.linalg.norm(errors, axis=1)
    metrics = {
        "RMSE_state": state_rmse(y_true, y_pred),
        "MAE_state": float(np.mean(state_abs_error)),
        "R2_mean": float(np.mean([r2_score(y_true[:, i], y_pred[:, i]) for i in range(3)])),
    }
    for i, axis_name in enumerate(STATE_COLS):
        metrics[f"RMSE_{axis_name}"] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        metrics[f"MAE_{axis_name}"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        metrics[f"R2_{axis_name}"] = float(r2_score(y_true[:, i], y_pred[:, i]))
    return metrics


def make_state_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=600,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def fit_state_models(X_train_scaled, y_train):
    fitted_models = {}
    for model_name, model in make_state_models().items():
        try:
            model.fit(X_train_scaled, y_train)
            fitted_models[model_name] = model
        except ValueError:
            wrapped_model = MultiOutputRegressor(model)
            wrapped_model.fit(X_train_scaled, y_train)
            fitted_models[model_name] = wrapped_model
    return fitted_models


def persistence_state_predict(X_values):
    return X_values.copy()


def train_test_state_split(dataset, train_fraction=0.8):
    split_idx = int(len(dataset) * train_fraction)
    X_values = dataset[FEATURE_COLS_STATE].to_numpy()
    y_values = dataset[TARGET_COLS_STATE].to_numpy()
    X_train, X_test = X_values[:split_idx], X_values[split_idx:]
    y_train, y_test = y_values[:split_idx], y_values[split_idx:]
    scaler_local = StandardScaler()
    X_train_scaled = scaler_local.fit_transform(X_train)
    X_test_scaled = scaler_local.transform(X_test)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler_local, split_idx


def evaluate_state_models_for_dataset(dataset, horizon, setting="same_trajectory"):
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler_local, split_idx = train_test_state_split(dataset)
    rows = []

    persistence_pred = persistence_state_predict(X_test)
    persistence_metrics = evaluate_state_prediction(y_test, persistence_pred)
    rows.append({
        "setting": setting,
        "horizon": horizon,
        "effective_time": float((t_eval[1] - t_eval[0]) * horizon),
        "model": "Persistence",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **persistence_metrics,
    })

    fitted_models = fit_state_models(X_train_scaled, y_train)
    for model_name, fitted_model in fitted_models.items():
        pred = fitted_model.predict(X_test_scaled)
        model_metrics = evaluate_state_prediction(y_test, pred)
        rows.append({
            "setting": setting,
            "horizon": horizon,
            "effective_time": float((t_eval[1] - t_eval[0]) * horizon),
            "model": model_name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            **model_metrics,
        })

    return rows, fitted_models, scaler_local, split_idx


print("增强实验公共函数已准备完成。")
# ---
horizon_rows = []
horizon_models = {}
horizon_scalers = {}

for horizon_k in HORIZONS:
    state_dataset = build_state_dataset(trajectory, horizon_k)
    rows, fitted_models, scaler_for_horizon, split_idx = evaluate_state_models_for_dataset(
        state_dataset,
        horizon=horizon_k,
        setting="same_trajectory",
    )
    horizon_rows.extend(rows)
    horizon_models[horizon_k] = fitted_models
    horizon_scalers[horizon_k] = scaler_for_horizon

horizon_results = pd.DataFrame(horizon_rows)
horizon_results.to_csv(RESULTS_DIR / "horizon_results.csv", index=False)

plt.figure(figsize=(9, 5))
sns.lineplot(
    data=horizon_results,
    x="horizon",
    y="RMSE_state",
    hue="model",
    marker="o",
)
plt.xlabel("Prediction horizon k")
plt.ylabel("State RMSE")
plt.title("Prediction horizon 对三维状态预测误差的影响")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "horizon_vs_rmse.png", dpi=300, bbox_inches="tight")
plt.show()

horizon_results[["horizon", "effective_time", "model", "RMSE_state", "MAE_state", "R2_mean"]]
# ---
def rollout_model(fitted_model, scaler_local, initial_state, steps):
    current_state = np.asarray(initial_state, dtype=float).reshape(1, -1)
    predictions = []
    for _ in range(steps):
        current_scaled = scaler_local.transform(current_state)
        next_state = fitted_model.predict(current_scaled)
        next_state = np.asarray(next_state).reshape(1, -1)
        predictions.append(next_state.ravel())
        current_state = next_state
    return np.asarray(predictions)


rollout_steps = 500
one_step_dataset = build_state_dataset(trajectory, horizon=1)
X_train_one, X_test_one, X_train_one_scaled, X_test_one_scaled, y_train_one, y_test_one, one_step_scaler, one_step_split_idx = train_test_state_split(one_step_dataset)
one_step_models = fit_state_models(X_train_one_scaled, y_train_one)

rollout_rows = []
rollout_predictions = {}
true_rollout = trajectory[STATE_COLS].iloc[one_step_split_idx + 1:one_step_split_idx + 1 + rollout_steps].to_numpy()
rollout_initial_state = trajectory[STATE_COLS].iloc[one_step_split_idx].to_numpy()

for model_name, fitted_model in one_step_models.items():
    predicted_rollout = rollout_model(fitted_model, one_step_scaler, rollout_initial_state, rollout_steps)
    rollout_predictions[model_name] = predicted_rollout
    step_errors = np.linalg.norm(true_rollout - predicted_rollout, axis=1)
    for step_idx, error_value in enumerate(step_errors, start=1):
        component_error = true_rollout[step_idx - 1] - predicted_rollout[step_idx - 1]
        rollout_rows.append({
            "model": model_name,
            "rollout_step": step_idx,
            "state_error": float(error_value),
            "squared_state_error": float(error_value ** 2),
            "abs_error_x": float(abs(component_error[0])),
            "abs_error_y": float(abs(component_error[1])),
            "abs_error_z": float(abs(component_error[2])),
            "true_x": float(true_rollout[step_idx - 1, 0]),
            "true_y": float(true_rollout[step_idx - 1, 1]),
            "true_z": float(true_rollout[step_idx - 1, 2]),
            "pred_x": float(predicted_rollout[step_idx - 1, 0]),
            "pred_y": float(predicted_rollout[step_idx - 1, 1]),
            "pred_z": float(predicted_rollout[step_idx - 1, 2]),
        })

rollout_results = pd.DataFrame(rollout_rows)
rollout_results["cumulative_RMSE_state"] = rollout_results.groupby("model")["squared_state_error"].transform(
    lambda values: np.sqrt(np.cumsum(values) / np.arange(1, len(values) + 1))
)
rollout_results.to_csv(RESULTS_DIR / "rollout_results.csv", index=False)

plt.figure(figsize=(9, 5))
sns.lineplot(
    data=rollout_results,
    x="rollout_step",
    y="cumulative_RMSE_state",
    hue="model",
)
plt.xlabel("Rollout step")
plt.ylabel("Cumulative state RMSE")
plt.title("Recursive rollout 误差累积")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rollout_error.png", dpi=300, bbox_inches="tight")
plt.show()

main_rollout_model = "MLP" if "MLP" in rollout_predictions else list(rollout_predictions.keys())[0]
main_predicted_rollout = rollout_predictions[main_rollout_model]

plt.figure(figsize=(10, 5))
plt.plot(true_rollout[:, 0], label="真实 x(t)", linewidth=1.5)
plt.plot(main_predicted_rollout[:, 0], label=f"{main_rollout_model} rollout 预测 x(t)", linewidth=1.2)
plt.xlabel("Rollout step")
plt.ylabel("x")
plt.title("Recursive rollout 中 x(t) 的真实值与预测值")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rollout_x_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(true_rollout[:, 0], true_rollout[:, 1], true_rollout[:, 2], label="真实轨迹", linewidth=1.0)
ax.plot(main_predicted_rollout[:, 0], main_predicted_rollout[:, 1], main_predicted_rollout[:, 2], label=f"{main_rollout_model} rollout", linewidth=1.0)
ax.set_title("真实轨迹与 recursive rollout 预测轨迹")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rollout_attractor.png", dpi=300, bbox_inches="tight")
plt.show()

rollout_results.groupby("model")[["state_error", "cumulative_RMSE_state"]].tail(1)
# ---
generalization_horizon = 10
base_state_dataset = build_state_dataset(trajectory, generalization_horizon)
X_train_base, X_test_base, X_train_base_scaled, X_test_base_scaled, y_train_base, y_test_base, base_scaler, base_split_idx = train_test_state_split(base_state_dataset)
base_models = fit_state_models(X_train_base_scaled, y_train_base)

generalization_rows = []

for initial_condition in GENERALIZATION_INITIAL_CONDITIONS:
    test_trajectory = simulate_lorenz(initial_condition)
    test_dataset = build_state_dataset(test_trajectory, generalization_horizon)
    X_generalization = test_dataset[FEATURE_COLS_STATE].to_numpy()
    y_generalization = test_dataset[TARGET_COLS_STATE].to_numpy()
    X_generalization_scaled = base_scaler.transform(X_generalization)
    ic_label = f"({initial_condition[0]}, {initial_condition[1]}, {initial_condition[2]})"

    persistence_pred = persistence_state_predict(X_generalization)
    persistence_metrics = evaluate_state_prediction(y_generalization, persistence_pred)
    generalization_rows.append({
        "initial_condition": ic_label,
        "horizon": generalization_horizon,
        "model": "Persistence",
        "test_rows": len(X_generalization),
        **persistence_metrics,
    })

    for model_name, fitted_model in base_models.items():
        pred = fitted_model.predict(X_generalization_scaled)
        model_metrics = evaluate_state_prediction(y_generalization, pred)
        generalization_rows.append({
            "initial_condition": ic_label,
            "horizon": generalization_horizon,
            "model": model_name,
            "test_rows": len(X_generalization),
            **model_metrics,
        })

initial_condition_generalization = pd.DataFrame(generalization_rows)
initial_condition_generalization.to_csv(RESULTS_DIR / "initial_condition_generalization.csv", index=False)

plt.figure(figsize=(10, 5))
sns.barplot(
    data=initial_condition_generalization,
    x="initial_condition",
    y="RMSE_state",
    hue="model",
)
plt.xlabel("Initial condition")
plt.ylabel("State RMSE")
plt.title("不同初始条件下的状态预测泛化误差")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "initial_condition_rmse.png", dpi=300, bbox_inches="tight")
plt.show()

initial_condition_generalization[["initial_condition", "model", "RMSE_state", "MAE_state", "R2_mean"]]
# ---
ENHANCEMENT_HORIZONS = [10, 100, 500]

MLP_ABLATION_CONFIGS = [
    {"name": "MLP-direct-scaled-relu-64x64", "mode": "direct", "hidden_layer_sizes": (64, 64), "activation": "relu"},
    {"name": "MLP-residual-relu-32x32", "mode": "residual", "hidden_layer_sizes": (32, 32), "activation": "relu"},
    {"name": "MLP-residual-relu-64x64", "mode": "residual", "hidden_layer_sizes": (64, 64), "activation": "relu"},
    {"name": "MLP-residual-relu-128x128", "mode": "residual", "hidden_layer_sizes": (128, 128), "activation": "relu"},
    {"name": "MLP-residual-relu-64x64x64", "mode": "residual", "hidden_layer_sizes": (64, 64, 64), "activation": "relu"},
    {"name": "MLP-residual-tanh-64x64", "mode": "residual", "hidden_layer_sizes": (64, 64), "activation": "tanh"},
]


def make_scaled_mlp(hidden_layer_sizes=(64, 64), activation="relu", random_state=RANDOM_STATE):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=random_state,
    )


def fit_scaled_mlp_state_model(X_train, X_train_scaled, y_train, config):
    y_scaler = StandardScaler()
    if config["mode"] == "residual":
        target_train = y_train - X_train
    else:
        target_train = y_train
    target_train_scaled = y_scaler.fit_transform(target_train)
    model = make_scaled_mlp(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config["activation"],
    )
    model.fit(X_train_scaled, target_train_scaled)
    return {"model": model, "y_scaler": y_scaler, "mode": config["mode"], "name": config["name"]}


def predict_scaled_mlp_state(fitted_bundle, X_values, X_scaled):
    pred_target_scaled = fitted_bundle["model"].predict(X_scaled)
    pred_target = fitted_bundle["y_scaler"].inverse_transform(pred_target_scaled)
    if fitted_bundle["mode"] == "residual":
        return X_values + pred_target
    return pred_target


class EchoStateRegressor:
    def __init__(self, n_reservoir=200, spectral_radius=0.9, input_scale=0.5, leak_rate=0.3, ridge=1e-5, random_state=RANDOM_STATE):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leak_rate = leak_rate
        self.ridge = ridge
        self.random_state = random_state

    def _initialize(self, input_dim):
        rng = np.random.default_rng(self.random_state)
        self.W_in_ = rng.uniform(-self.input_scale, self.input_scale, size=(self.n_reservoir, input_dim + 1))
        W = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_reservoir))
        eig_radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W_ = W * (self.spectral_radius / eig_radius)

    def _step(self, input_vector, reservoir_state):
        augmented_input = np.concatenate(([1.0], input_vector))
        pre_activation = self.W_in_ @ augmented_input + self.W_ @ reservoir_state
        updated = np.tanh(pre_activation)
        return (1.0 - self.leak_rate) * reservoir_state + self.leak_rate * updated

    def _collect_states(self, X):
        reservoir_state = np.zeros(self.n_reservoir)
        states = []
        for input_vector in X:
            reservoir_state = self._step(input_vector, reservoir_state)
            states.append(reservoir_state.copy())
        return np.asarray(states)

    def fit(self, X, y):
        self._initialize(X.shape[1])
        states = self._collect_states(X)
        design = np.hstack([np.ones((len(X), 1)), X, states])
        regularizer = self.ridge * np.eye(design.shape[1])
        self.W_out_ = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
        return self

    def predict(self, X):
        states = self._collect_states(X)
        design = np.hstack([np.ones((len(X), 1)), X, states])
        return design @ self.W_out_

    def step_predict(self, input_vector, reservoir_state):
        next_reservoir_state = self._step(input_vector, reservoir_state)
        design = np.concatenate(([1.0], input_vector, next_reservoir_state))
        return design @ self.W_out_, next_reservoir_state


def fit_esn_residual_model(X_train, X_train_scaled, y_train):
    delta_scaler = StandardScaler()
    delta_train_scaled = delta_scaler.fit_transform(y_train - X_train)
    esn_model = EchoStateRegressor()
    esn_model.fit(X_train_scaled, delta_train_scaled)
    return {"model": esn_model, "delta_scaler": delta_scaler, "mode": "residual", "name": "ESN-residual"}


def predict_esn_residual(bundle, X_values, X_scaled):
    delta_scaled = bundle["model"].predict(X_scaled)
    delta = bundle["delta_scaler"].inverse_transform(delta_scaled)
    return X_values + delta


print("模型增强函数已准备完成。")
# ---
model_enhancement_rows = []

for horizon_k in ENHANCEMENT_HORIZONS:
    state_dataset = build_state_dataset(trajectory, horizon_k)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler_local, split_idx = train_test_state_split(state_dataset)

    baseline_subset = horizon_results[
        (horizon_results["horizon"] == horizon_k)
        & (horizon_results["model"].isin(["Random Forest", "MLP"]))
    ].copy()
    for _, row in baseline_subset.iterrows():
        model_enhancement_rows.append({
            "horizon": horizon_k,
            "model": f"baseline-{row['model']}",
            "family": "baseline",
            "RMSE_state": row["RMSE_state"],
            "MAE_state": row["MAE_state"],
            "R2_mean": row["R2_mean"],
            "RMSE_x": row["RMSE_x"],
            "RMSE_y": row["RMSE_y"],
            "RMSE_z": row["RMSE_z"],
        })

    for config in MLP_ABLATION_CONFIGS:
        fitted_bundle = fit_scaled_mlp_state_model(X_train, X_train_scaled, y_train, config)
        pred = predict_scaled_mlp_state(fitted_bundle, X_test, X_test_scaled)
        metrics = evaluate_state_prediction(y_test, pred)
        model_enhancement_rows.append({
            "horizon": horizon_k,
            "model": config["name"],
            "family": "MLP",
            **metrics,
        })

    esn_bundle = fit_esn_residual_model(X_train, X_train_scaled, y_train)
    esn_pred = predict_esn_residual(esn_bundle, X_test, X_test_scaled)
    esn_metrics = evaluate_state_prediction(y_test, esn_pred)
    model_enhancement_rows.append({
        "horizon": horizon_k,
        "model": "ESN-residual",
        "family": "ESN",
        **esn_metrics,
    })

model_enhancement_results = pd.DataFrame(model_enhancement_rows)
model_enhancement_results.to_csv(RESULTS_DIR / "model_enhancement_results.csv", index=False)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=model_enhancement_results,
    x="horizon",
    y="RMSE_state",
    hue="model",
)
plt.xlabel("Prediction horizon k")
plt.ylabel("State RMSE")
plt.title("模型增强方法在不同 horizon 下的三维状态预测误差")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_enhancement_horizon_rmse.png", dpi=300, bbox_inches="tight")
plt.show()

model_enhancement_results[["horizon", "model", "RMSE_state", "R2_mean"]].sort_values(["horizon", "RMSE_state"])
# ---
def rollout_scaled_mlp_bundle(fitted_bundle, x_scaler, initial_state, steps):
    current_state = np.asarray(initial_state, dtype=float).reshape(1, -1)
    predictions = []
    for _ in range(steps):
        current_scaled = x_scaler.transform(current_state)
        next_state = predict_scaled_mlp_state(fitted_bundle, current_state, current_scaled)
        predictions.append(next_state.ravel())
        current_state = next_state.reshape(1, -1)
    return np.asarray(predictions)


def rollout_esn_bundle(esn_bundle, x_scaler, initial_state, steps):
    current_state = np.asarray(initial_state, dtype=float).reshape(1, -1)
    reservoir_state = np.zeros(esn_bundle["model"].n_reservoir)
    predictions = []
    for _ in range(steps):
        current_scaled = x_scaler.transform(current_state).ravel()
        delta_scaled, reservoir_state = esn_bundle["model"].step_predict(current_scaled, reservoir_state)
        delta = esn_bundle["delta_scaler"].inverse_transform(delta_scaled.reshape(1, -1))
        next_state = current_state + delta
        predictions.append(next_state.ravel())
        current_state = next_state
    return np.asarray(predictions)


enhanced_rollout_steps = 500
one_step_dataset = build_state_dataset(trajectory, horizon=1)
X_train_one, X_test_one, X_train_one_scaled, X_test_one_scaled, y_train_one, y_test_one, enhanced_one_scaler, enhanced_one_split_idx = train_test_state_split(one_step_dataset)
enhanced_true_rollout = trajectory[STATE_COLS].iloc[enhanced_one_split_idx + 1:enhanced_one_split_idx + 1 + enhanced_rollout_steps].to_numpy()
enhanced_initial_state = trajectory[STATE_COLS].iloc[enhanced_one_split_idx].to_numpy()

rollout_model_configs = [
    {"name": "MLP-residual-relu-64x64", "mode": "residual", "hidden_layer_sizes": (64, 64), "activation": "relu"},
    {"name": "MLP-residual-tanh-64x64", "mode": "residual", "hidden_layer_sizes": (64, 64), "activation": "tanh"},
    {"name": "MLP-residual-relu-128x128", "mode": "residual", "hidden_layer_sizes": (128, 128), "activation": "relu"},
]

enhanced_rollout_rows = []
enhanced_rollout_predictions = {}

for config in rollout_model_configs:
    fitted_bundle = fit_scaled_mlp_state_model(X_train_one, X_train_one_scaled, y_train_one, config)
    predicted_rollout = rollout_scaled_mlp_bundle(fitted_bundle, enhanced_one_scaler, enhanced_initial_state, enhanced_rollout_steps)
    enhanced_rollout_predictions[config["name"]] = predicted_rollout
    step_errors = np.linalg.norm(enhanced_true_rollout - predicted_rollout, axis=1)
    for step_idx, error_value in enumerate(step_errors, start=1):
        enhanced_rollout_rows.append({
            "model": config["name"],
            "rollout_step": step_idx,
            "state_error": float(error_value),
            "squared_state_error": float(error_value ** 2),
            "true_x": float(enhanced_true_rollout[step_idx - 1, 0]),
            "pred_x": float(predicted_rollout[step_idx - 1, 0]),
        })

esn_one_bundle = fit_esn_residual_model(X_train_one, X_train_one_scaled, y_train_one)
esn_rollout = rollout_esn_bundle(esn_one_bundle, enhanced_one_scaler, enhanced_initial_state, enhanced_rollout_steps)
enhanced_rollout_predictions["ESN-residual"] = esn_rollout
esn_step_errors = np.linalg.norm(enhanced_true_rollout - esn_rollout, axis=1)
for step_idx, error_value in enumerate(esn_step_errors, start=1):
    enhanced_rollout_rows.append({
        "model": "ESN-residual",
        "rollout_step": step_idx,
        "state_error": float(error_value),
        "squared_state_error": float(error_value ** 2),
        "true_x": float(enhanced_true_rollout[step_idx - 1, 0]),
        "pred_x": float(esn_rollout[step_idx - 1, 0]),
    })

enhanced_rollout_results = pd.DataFrame(enhanced_rollout_rows)
enhanced_rollout_results["cumulative_RMSE_state"] = enhanced_rollout_results.groupby("model")["squared_state_error"].transform(
    lambda values: np.sqrt(np.cumsum(values) / np.arange(1, len(values) + 1))
)
enhanced_rollout_results.to_csv(RESULTS_DIR / "enhanced_rollout_results.csv", index=False)

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=enhanced_rollout_results,
    x="rollout_step",
    y="cumulative_RMSE_state",
    hue="model",
)
plt.xlabel("Rollout step")
plt.ylabel("Cumulative state RMSE")
plt.title("Residual MLP 与 ESN 的 recursive rollout 误差")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "enhanced_rollout_error.png", dpi=300, bbox_inches="tight")
plt.show()

best_enhanced_rollout_name = enhanced_rollout_results.groupby("model")["cumulative_RMSE_state"].last().idxmin()
best_enhanced_rollout = enhanced_rollout_predictions[best_enhanced_rollout_name]

plt.figure(figsize=(10, 5))
plt.plot(enhanced_true_rollout[:, 0], label="真实 x(t)", linewidth=1.5)
plt.plot(best_enhanced_rollout[:, 0], label=f"{best_enhanced_rollout_name} 预测 x(t)", linewidth=1.2)
plt.xlabel("Rollout step")
plt.ylabel("x")
plt.title("增强模型 rollout 中 x(t) 的真实值与预测值")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "enhanced_rollout_x_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

enhanced_rollout_results.groupby("model")[["state_error", "cumulative_RMSE_state"]].tail(1).sort_values("cumulative_RMSE_state")
# ---
def build_sindy_library(states):
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    theta = np.column_stack([
        np.ones(len(states)),
        x,
        y,
        z,
        x ** 2,
        x * y,
        x * z,
        y ** 2,
        y * z,
        z ** 2,
    ])
    feature_names = ["1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2"]
    return theta, feature_names


def sequential_thresholded_least_squares(theta, derivatives, threshold=0.05, max_iter=12):
    coefficients = np.linalg.lstsq(theta, derivatives, rcond=None)[0]
    for _ in range(max_iter):
        small = np.abs(coefficients) < threshold
        coefficients[small] = 0.0
        for target_idx in range(derivatives.shape[1]):
            active = ~small[:, target_idx]
            if np.any(active):
                coefficients[active, target_idx] = np.linalg.lstsq(
                    theta[:, active], derivatives[:, target_idx], rcond=None
                )[0]
    coefficients[np.abs(coefficients) < threshold] = 0.0
    return coefficients


def format_sindy_equation(coefficients, feature_names, target_name, precision=4):
    terms = []
    for coef, feature in zip(coefficients, feature_names):
        if abs(coef) > 1e-10:
            sign = "+" if coef >= 0 else "-"
            terms.append(f" {sign} {abs(coef):.{precision}f}{feature}")
    if not terms:
        rhs = "0"
    else:
        rhs = "".join(terms).lstrip(" +")
    return f"d{target_name}/dt = {rhs}"


def sindy_predict_derivative(state, coefficients):
    theta, _ = build_sindy_library(np.asarray(state, dtype=float).reshape(1, -1))
    return (theta @ coefficients).ravel()


def rk4_step_vector_field(state, dt, derivative_func):
    state = np.asarray(state, dtype=float)
    k1 = derivative_func(state)
    k2 = derivative_func(state + 0.5 * dt * k1)
    k3 = derivative_func(state + 0.5 * dt * k2)
    k4 = derivative_func(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout_vector_field(initial_state, steps, dt, derivative_func):
    current_state = np.asarray(initial_state, dtype=float)
    predictions = []
    for _ in range(steps):
        current_state = rk4_step_vector_field(current_state, dt, derivative_func)
        predictions.append(current_state.copy())
    return np.asarray(predictions)

states = trajectory[STATE_COLS].to_numpy()
dt = float(t_eval[1] - t_eval[0])
centered_states = states[1:-1]
centered_derivatives = (states[2:] - states[:-2]) / (2 * dt)

sindy_theta, sindy_feature_names = build_sindy_library(centered_states)
sindy_coefficients = sequential_thresholded_least_squares(
    sindy_theta,
    centered_derivatives,
    threshold=0.05,
)

sindy_coefficients_df = pd.DataFrame(
    sindy_coefficients,
    index=sindy_feature_names,
    columns=["dx_dt", "dy_dt", "dz_dt"],
)
sindy_coefficients_df.to_csv(RESULTS_DIR / "sindy_coefficients.csv")

true_coefficients = pd.DataFrame(0.0, index=sindy_feature_names, columns=["dx_dt", "dy_dt", "dz_dt"])
true_coefficients.loc["x", "dx_dt"] = -sigma
true_coefficients.loc["y", "dx_dt"] = sigma
true_coefficients.loc["x", "dy_dt"] = rho
true_coefficients.loc["y", "dy_dt"] = -1.0
true_coefficients.loc["xz", "dy_dt"] = -1.0
true_coefficients.loc["xy", "dz_dt"] = 1.0
true_coefficients.loc["z", "dz_dt"] = -beta

sindy_equation_error = (sindy_coefficients_df - true_coefficients).rename_axis("feature").reset_index()
sindy_equation_error_long = sindy_equation_error.melt(
    id_vars="feature",
    var_name="equation",
    value_name="coefficient_error",
)
sindy_equation_error_long.to_csv(RESULTS_DIR / "sindy_equation_error.csv", index=False)

plt.figure(figsize=(8, 5))
sns.heatmap(sindy_coefficients_df, annot=True, fmt=".3f", cmap="coolwarm", center=0)
plt.title("SINDy 恢复的 Lorenz 方程系数")
plt.ylabel("候选项")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sindy_coefficients.png", dpi=300, bbox_inches="tight")
plt.show()

sindy_rollout_steps = 500
sindy_initial_index = int(len(trajectory) * 0.8)
sindy_initial_state = trajectory[STATE_COLS].iloc[sindy_initial_index].to_numpy()
sindy_true_rollout = trajectory[STATE_COLS].iloc[sindy_initial_index + 1:sindy_initial_index + 1 + sindy_rollout_steps].to_numpy()
sindy_rollout = rollout_vector_field(
    sindy_initial_state,
    sindy_rollout_steps,
    dt,
    lambda state: sindy_predict_derivative(state, sindy_coefficients),
)

plt.figure(figsize=(10, 5))
plt.plot(sindy_true_rollout[:, 0], label="真实 x(t)", linewidth=1.5)
plt.plot(sindy_rollout[:, 0], label="SINDy rollout x(t)", linewidth=1.2)
plt.xlabel("Rollout step")
plt.ylabel("x")
plt.title("SINDy 方程递归推演中的 x(t) 对比")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sindy_rollout_x_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

for axis_name, equation_col in zip(STATE_COLS, ["dx_dt", "dy_dt", "dz_dt"]):
    print(format_sindy_equation(sindy_coefficients_df[equation_col].to_numpy(), sindy_feature_names, axis_name))

sindy_coefficients_df

# ---
IMPERFECT_RHO = 26.0
LORENZ_MAX_LYAPUNOV = 0.9
LYAPUNOV_TIME = 1.0 / LORENZ_MAX_LYAPUNOV


def lorenz_derivative_array(state, sigma_value=sigma, rho_value=rho, beta_value=beta):
    x, y, z = np.asarray(state, dtype=float)
    return np.array([
        sigma_value * (y - x),
        x * (rho_value - z) - y,
        x * y - beta_value * z,
    ])


def imperfect_physics_step(state, dt=dt):
    return rk4_step_vector_field(
        state,
        dt,
        lambda s: lorenz_derivative_array(s, sigma_value=sigma, rho_value=IMPERFECT_RHO, beta_value=beta),
    )


def batch_imperfect_physics_predict(states_array):
    return np.asarray([imperfect_physics_step(row) for row in states_array])


def fit_hybrid_residual_mlp(X_train_scaled, residual_train):
    residual_scaler = StandardScaler()
    residual_train_scaled = residual_scaler.fit_transform(residual_train)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=600,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, residual_train_scaled)
    return {"model": model, "residual_scaler": residual_scaler, "name": "Hybrid MLP"}


def predict_hybrid_mlp(bundle, physics_prediction, X_scaled):
    residual_scaled = bundle["model"].predict(X_scaled)
    residual = bundle["residual_scaler"].inverse_transform(residual_scaled)
    return physics_prediction + residual


def rollout_pure_mlp(fitted_bundle, x_scaler, initial_state, steps):
    current_state = np.asarray(initial_state, dtype=float).reshape(1, -1)
    predictions = []
    for _ in range(steps):
        current_scaled = x_scaler.transform(current_state)
        next_state = predict_scaled_mlp_state(fitted_bundle, current_state, current_scaled)
        predictions.append(next_state.ravel())
        current_state = next_state.reshape(1, -1)
    return np.asarray(predictions)


def rollout_hybrid_rf(model, x_scaler, initial_state, steps):
    current_state = np.asarray(initial_state, dtype=float).reshape(1, -1)
    predictions = []
    for _ in range(steps):
        physics_next = batch_imperfect_physics_predict(current_state)
        residual = model.predict(x_scaler.transform(current_state))
        next_state = physics_next + residual
        predictions.append(next_state.ravel())
        current_state = next_state.reshape(1, -1)
    return np.asarray(predictions)


def rollout_hybrid_mlp(bundle, x_scaler, initial_state, steps):
    current_state = np.asarray(initial_state, dtype=float).reshape(1, -1)
    predictions = []
    for _ in range(steps):
        physics_next = batch_imperfect_physics_predict(current_state)
        next_state = predict_hybrid_mlp(bundle, physics_next, x_scaler.transform(current_state))
        predictions.append(next_state.ravel())
        current_state = next_state.reshape(1, -1)
    return np.asarray(predictions)


def first_valid_prediction_failure(errors, threshold):
    exceeded = np.flatnonzero(errors > threshold)
    if len(exceeded) == 0:
        return len(errors), float(len(errors) * dt)
    valid_steps = int(exceeded[0])
    return valid_steps, float(valid_steps * dt)


def summarize_long_term_statistics(model_name, rollout_values):
    rows = []
    for idx, axis_name in enumerate(STATE_COLS):
        values = rollout_values[:, idx]
        rows.append({
            "model": model_name,
            "axis": axis_name,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return rows

hybrid_one_step_dataset = build_state_dataset(trajectory, horizon=1)
X_train_hybrid, X_test_hybrid, X_train_hybrid_scaled, X_test_hybrid_scaled, y_train_hybrid, y_test_hybrid, hybrid_x_scaler, hybrid_split_idx = train_test_state_split(hybrid_one_step_dataset)

physics_train_pred = batch_imperfect_physics_predict(X_train_hybrid)
physics_test_pred = batch_imperfect_physics_predict(X_test_hybrid)
residual_train = y_train_hybrid - physics_train_pred

hybrid_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
hybrid_rf.fit(X_train_hybrid_scaled, residual_train)
hybrid_rf_pred = physics_test_pred + hybrid_rf.predict(X_test_hybrid_scaled)

hybrid_mlp = fit_hybrid_residual_mlp(X_train_hybrid_scaled, residual_train)
hybrid_mlp_pred = predict_hybrid_mlp(hybrid_mlp, physics_test_pred, X_test_hybrid_scaled)

pure_mlp_config = {"name": "Pure ML Residual MLP", "mode": "residual", "hidden_layer_sizes": (64, 64), "activation": "tanh"}
pure_mlp_bundle = fit_scaled_mlp_state_model(X_train_hybrid, X_train_hybrid_scaled, y_train_hybrid, pure_mlp_config)
pure_mlp_pred = predict_scaled_mlp_state(pure_mlp_bundle, X_test_hybrid, X_test_hybrid_scaled)

hybrid_metric_predictions = {
    "Imperfect physics": physics_test_pred,
    "Pure ML Residual MLP": pure_mlp_pred,
    "Hybrid RF": hybrid_rf_pred,
    "Hybrid MLP": hybrid_mlp_pred,
}
hybrid_metric_rows = []
for model_name, prediction_values in hybrid_metric_predictions.items():
    hybrid_metric_rows.append({"model": model_name, **evaluate_state_prediction(y_test_hybrid, prediction_values)})
hybrid_metrics = pd.DataFrame(hybrid_metric_rows)
hybrid_metrics.to_csv(RESULTS_DIR / "hybrid_metrics.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=hybrid_metrics, x="model", y="RMSE_state")
plt.ylabel("One-step state RMSE")
plt.xlabel("")
plt.title("不完美物理模型、Pure ML 与 Hybrid correction 的一步预测误差")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hybrid_one_step_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

hybrid_rollout_steps = 500
hybrid_initial_state = trajectory[STATE_COLS].iloc[hybrid_split_idx].to_numpy()
hybrid_true_rollout = trajectory[STATE_COLS].iloc[hybrid_split_idx + 1:hybrid_split_idx + 1 + hybrid_rollout_steps].to_numpy()

hybrid_rollout_predictions = {
    "Imperfect physics": rollout_vector_field(
        hybrid_initial_state,
        hybrid_rollout_steps,
        dt,
        lambda s: lorenz_derivative_array(s, sigma_value=sigma, rho_value=IMPERFECT_RHO, beta_value=beta),
    ),
    "Pure ML Residual MLP": rollout_pure_mlp(pure_mlp_bundle, hybrid_x_scaler, hybrid_initial_state, hybrid_rollout_steps),
    "Hybrid RF": rollout_hybrid_rf(hybrid_rf, hybrid_x_scaler, hybrid_initial_state, hybrid_rollout_steps),
    "Hybrid MLP": rollout_hybrid_mlp(hybrid_mlp, hybrid_x_scaler, hybrid_initial_state, hybrid_rollout_steps),
}

hybrid_rollout_rows = []
for model_name, prediction_values in hybrid_rollout_predictions.items():
    step_errors = np.linalg.norm(hybrid_true_rollout - prediction_values, axis=1)
    for step_idx, error_value in enumerate(step_errors, start=1):
        hybrid_rollout_rows.append({
            "model": model_name,
            "rollout_step": step_idx,
            "physical_time": float(step_idx * dt),
            "lyapunov_time": float((step_idx * dt) / LYAPUNOV_TIME),
            "state_error": float(error_value),
            "squared_state_error": float(error_value ** 2),
            "true_x": float(hybrid_true_rollout[step_idx - 1, 0]),
            "pred_x": float(prediction_values[step_idx - 1, 0]),
        })

hybrid_rollout_results = pd.DataFrame(hybrid_rollout_rows)
hybrid_rollout_results["cumulative_RMSE_state"] = hybrid_rollout_results.groupby("model")["squared_state_error"].transform(
    lambda values: np.sqrt(np.cumsum(values) / np.arange(1, len(values) + 1))
)
hybrid_rollout_results.to_csv(RESULTS_DIR / "hybrid_rollout_results.csv", index=False)

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=hybrid_rollout_results,
    x="rollout_step",
    y="cumulative_RMSE_state",
    hue="model",
)
plt.xlabel("Rollout step")
plt.ylabel("Cumulative state RMSE")
plt.title("Hybrid correction 的 recursive rollout 误差")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hybrid_rollout_error.png", dpi=300, bbox_inches="tight")
plt.show()

valid_threshold = float(0.4 * np.linalg.norm(np.std(X_train_hybrid, axis=0)))
valid_prediction_rows = []
for model_name, prediction_values in hybrid_rollout_predictions.items():
    step_errors = np.linalg.norm(hybrid_true_rollout - prediction_values, axis=1)
    valid_steps, valid_time = first_valid_prediction_failure(step_errors, valid_threshold)
    valid_prediction_rows.append({
        "model": model_name,
        "threshold": valid_threshold,
        "valid_steps": valid_steps,
        "valid_physical_time": valid_time,
        "valid_lyapunov_time": float(valid_time / LYAPUNOV_TIME),
        "final_cumulative_RMSE_state": float(
            hybrid_rollout_results.loc[hybrid_rollout_results["model"] == model_name, "cumulative_RMSE_state"].iloc[-1]
        ),
    })
valid_prediction_time = pd.DataFrame(valid_prediction_rows)
valid_prediction_time.to_csv(RESULTS_DIR / "valid_prediction_time.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=valid_prediction_time, x="model", y="valid_lyapunov_time")
plt.ylabel("Valid prediction time / Lyapunov time")
plt.xlabel("")
plt.title("不同模型的有效预测时间")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "valid_prediction_time.png", dpi=300, bbox_inches="tight")
plt.show()

long_term_statistics_rows = summarize_long_term_statistics("True trajectory", hybrid_true_rollout)
for model_name, prediction_values in hybrid_rollout_predictions.items():
    long_term_statistics_rows.extend(summarize_long_term_statistics(model_name, prediction_values))
long_term_statistics = pd.DataFrame(long_term_statistics_rows)
long_term_statistics.to_csv(RESULTS_DIR / "long_term_statistics.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, axis_idx, axis_name in zip(axes, range(3), STATE_COLS):
    sns.kdeplot(hybrid_true_rollout[:, axis_idx], ax=ax, label="True", linewidth=1.7)
    for model_name in ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"]:
        sns.kdeplot(hybrid_rollout_predictions[model_name][:, axis_idx], ax=ax, label=model_name, linewidth=1.0, alpha=0.85)
    ax.set_title(f"{axis_name} distribution")
    ax.set_xlabel(axis_name)
axes[0].legend(fontsize=8)
plt.suptitle("真实轨迹与 rollout 轨迹的长期分布对比")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "long_term_distribution_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Valid prediction threshold: {valid_threshold:.4f}")
display(hybrid_metrics)
display(valid_prediction_time)
long_term_statistics.head(12)

# ---
