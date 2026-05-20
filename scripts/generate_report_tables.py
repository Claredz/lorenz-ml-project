from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
ROW_END = r" \\"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS_DIR / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def fmt(value: str | float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def find_row(data: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    for row in data:
        if all(row.get(key, row.get("", "")) == value for key, value in conditions.items()):
            return row
    raise KeyError(conditions)


def section(title: str, rows: list[str]) -> str:
    body = "\n".join(rows)
    return f"% {title}\n{body}\n"


def original_metrics() -> str:
    labels = {
        "Linear Regression": "Linear Regression",
        "Random Forest": "Random Forest",
        "MLP": "MLP",
    }
    data = read_csv("model_metrics.csv")
    rows = []
    for model in ["Linear Regression", "Random Forest", "MLP"]:
        row = find_row(data, **{"": model})
        rows.append(f"{labels[model]} & {fmt(row['RMSE'])} & {fmt(row['MAE'])} & {fmt(row['R2'])}{ROW_END}")
    return section("Direct prediction metrics from results/model_metrics.csv", rows)


def horizon_metrics() -> str:
    data = read_csv("horizon_results.csv")
    wanted = [
        ("10", "Linear Regression"),
        ("10", "Random Forest"),
        ("10", "MLP"),
        ("100", "Linear Regression"),
        ("100", "Random Forest"),
        ("100", "MLP"),
        ("500", "Linear Regression"),
        ("500", "Random Forest"),
        ("500", "MLP"),
    ]
    rows = []
    for horizon, model in wanted:
        row = find_row(data, horizon=horizon, model=model)
        rows.append(f"{horizon} & {model} & {fmt(row['RMSE_state'])} & {fmt(row['R2_mean'])}{ROW_END}")
    return section("Representative horizon metrics from results/horizon_results.csv", rows)


def residual_metrics() -> str:
    data = read_csv("model_enhancement_results.csv")
    wanted = [
        ("10", "baseline-MLP", "Direct MLP"),
        ("10", "MLP-residual-relu-64x64x64", "Residual MLP"),
        ("100", "baseline-MLP", "Direct MLP"),
        ("100", "MLP-residual-relu-64x64x64", "Residual MLP"),
        ("500", "baseline-MLP", "Direct MLP"),
        ("500", "MLP-residual-relu-64x64x64", "Residual MLP"),
    ]
    rows = []
    for horizon, model, label in wanted:
        row = find_row(data, horizon=horizon, model=model)
        rows.append(f"{horizon} & {label} & {fmt(row['RMSE_state'])} & {fmt(row['R2_mean'])}{ROW_END}")
    return section("Direct MLP vs Residual MLP from results/model_enhancement_results.csv", rows)


def rollout_summary() -> str:
    data = read_csv("rollout_results.csv")
    latest: dict[str, dict[str, str]] = {}
    for row in data:
        latest[row["model"]] = row
    rows = []
    for model in ["Linear Regression", "Random Forest", "MLP"]:
        row = latest[model]
        rows.append(f"{model} & {int(float(row['rollout_step']))} & {fmt(row['cumulative_RMSE_state'])}{ROW_END}")
    return section("Rollout final cumulative RMSE from results/rollout_results.csv", rows)


def hybrid_metrics() -> str:
    data = read_csv("hybrid_metrics.csv")
    rows = []
    for model in ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"]:
        row = find_row(data, model=model)
        rows.append(f"{model} & {fmt(row['RMSE_state'])} & {fmt(row['R2_mean'], 10)}{ROW_END}")
    return section("Hybrid one-step metrics from results/hybrid_metrics.csv", rows)


def valid_prediction_time() -> str:
    data = read_csv("valid_prediction_time.csv")
    rows = []
    for model in ["Imperfect physics", "Pure ML Residual MLP", "Hybrid RF", "Hybrid MLP"]:
        row = find_row(data, model=model)
        rows.append(
            f"{model} & {int(float(row['valid_steps']))} & {fmt(row['valid_physical_time'])} & "
            f"{fmt(row['valid_lyapunov_time'])} & {fmt(row['final_cumulative_RMSE_state'])}{ROW_END}"
        )
    return section("Valid prediction time from results/valid_prediction_time.csv", rows)


def build() -> str:
    return "\n".join([
        original_metrics(),
        horizon_metrics(),
        rollout_summary(),
        residual_metrics(),
        hybrid_metrics(),
        valid_prediction_time(),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table rows from results CSV files.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "generated_report_tables.tex")
    args = parser.parse_args()
    content = build()
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
