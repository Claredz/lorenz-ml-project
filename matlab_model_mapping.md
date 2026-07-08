# MATLAB 模型对应表

| 论文模型名 | 原 Python 文件 | MATLAB 实现文件 | 是否真实训练 / 计算 | 输出 CSV | report.tex 是否引用该 CSV |
|---|---|---|---|---|---|
| Linear Regression | `scratch_code.py` | `matlab_src/runLorenz63Experiments.m` | 是，最小二乘训练 | `results/model_metrics.csv`, `results/horizon_results.csv`, `results/rollout_results.csv` | 是 |
| Random Forest | `scratch_code.py` | `matlab_src/runLorenz63Experiments.m` | 是，`TreeBagger` bagged regression trees | `results/model_metrics.csv`, `results/horizon_results.csv`, `results/rollout_results.csv` | 是 |
| MLP | `scratch_code.py` | `matlab_src/runLorenz63Experiments.m` | 是，`fitrnet` 训练 | `results/model_metrics.csv`, `results/horizon_results.csv`, `results/rollout_results.csv` | 是 |
| Residual MLP | `scratch_code.py` | `matlab_src/runLorenz63Experiments.m` | 是，`fitrnet` 残差目标训练 | `results/model_enhancement_results.csv` | 间接用于可复现表格核对 |
| Hybrid correction | `scratch_code.py` | `matlab_src/runLorenz63Experiments.m` | 是，不完美物理步 + 残差模型训练 | `results/hybrid_metrics.csv`, `results/valid_prediction_time.csv` | 间接用于可复现表格核对 |
| Baseline LSTM | `lorenz96_lstm.py`, `lorenz96_lstm_optim.py` | `matlab_src/runLorenz96Experiments.m` | 是，MATLAB LSTM 绝对状态预测训练 | `results/lstm_baseline_rollout.csv`, `results/lstm_phase_summary.csv`, `results/advanced/Baseline_LSTM_rollout.csv` | 是 |
| Phase 1 Residual LSTM + LayerNorm + gradient clipping | `lorenz96_lstm_optim.py` | `matlab_src/runLorenz96Experiments.m` | 是，MATLAB LSTM 残差训练 | `results/lstm_phase1_rollout.csv`, `results/lstm_phase_summary.csv` | 是 |
| Phase 2 Scheduled Sampling LSTM | `lorenz96_lstm_phase2.py` | `matlab_src/runLorenz96Experiments.m` | 是，按原线性 teacher-forcing 公式构造 scheduled-sampling 训练样本并训练 LSTM | `results/lstm_phase2_rollout.csv`, `results/lstm_phase_summary.csv` | 是 |
| Phase 3 window ablation w=10 | `lorenz96_lstm_phase3.py` | `matlab_src/runLorenz96Experiments.m` | 是，窗口 `w=10` 残差 LSTM 训练 | `results/lstm_phase3_w10_rollout.csv`, `results/lstm_phase3_summary.csv` | 是 |
| Phase 3 window ablation w=20 | `lorenz96_lstm_phase3.py` | `matlab_src/runLorenz96Experiments.m` | 是，窗口 `w=20` 残差 LSTM 训练 | `results/lstm_phase3_w20_rollout.csv`, `results/lstm_phase3_summary.csv` | 是 |
| Phase 3 window ablation w=50 | `lorenz96_lstm_phase3.py` | `matlab_src/runLorenz96Experiments.m` | 是，窗口 `w=50` 残差 LSTM 训练 | `results/lstm_phase3_w50_rollout.csv`, `results/lstm_phase3_summary.csv` | 是 |
| PINN LSTM | `lorenz96_advanced_experiments.py` | `matlab_src/runLorenz96Experiments.m` | 是，LSTM 训练目标加入 `lambda=0.1` 物理一致性目标 | `results/advanced/PINN_LSTM_rollout.csv`, `results/advanced/summary.csv` | 是 |
| Refined SS LSTM | `lorenz96_advanced_experiments.py` | `matlab_src/runLorenz96Experiments.m` | 是，按原 inverse-sigmoid teacher-forcing 公式构造训练样本并训练 LSTM | `results/advanced/Refined_SS_LSTM_rollout.csv`, `results/advanced/summary.csv` | 是 |
| Transformer | `lorenz96_advanced_experiments.py` | `matlab_src/runLorenz96Experiments.m` | 是，MATLAB self-attention 网络训练 | `results/advanced/Transformer_rollout.csv`, `results/advanced/summary.csv` | 是 |
| Hybrid LSTM | `lorenz96_advanced_experiments.py` | `matlab_src/runLorenz96Experiments.m` | 是，不完美 RK4 (`F_imperfect=8.5`) + LSTM 残差训练 | `results/advanced/Hybrid_LSTM_rollout.csv`, `results/advanced/summary.csv` | 是 |
| Ultimate Hybrid | `lorenz96_advanced_experiments.py` | `matlab_src/runLorenz96Experiments.m` | 是，不完美 RK4 + self-attention 残差 + PINN 目标 + refined scheduled sampling | `results/advanced/Ultimate_Hybrid_rollout.csv`, `results/advanced/summary.csv` | 是 |
