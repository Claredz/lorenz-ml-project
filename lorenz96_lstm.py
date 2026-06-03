"""
Lorenz-96 (10D) System LSTM Optimization Experiment

This script implements the final optimized LSTM architecture for predicting 
the Lorenz-96 chaotic system (10 dimensions).

Optimization Highlights:
1. Residual Learning: Predicts state increment (Delta x) instead of absolute state.
2. Stability: Uses LayerNorm and Gradient Clipping.
3. Hyperparameters: Window size = 20.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ==========================================
# Configurations
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

N = 10              # Lorenz-96 dimension
F = 8.0             # Forcing constant
WINDOW_SIZE = 20    # Optimized sequence window
EPOCHS = 20
BATCH_SIZE = 64
HIDDEN_DIM = 64
LR = 1e-3

# ==========================================
# 1. Lorenz-96 Data Generation
# ==========================================
def lorenz96(t, x):
    """Lorenz-96 model with periodic boundary conditions"""
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i+1)%N] - x[i-2]) * x[i-1] - x[i] + F
    return dxdt

print(f"Generating Lorenz-96 data (Dimension={N})...")
t_start, t_end = 0, 100
n_points = 10000
dt = (t_end - t_start) / n_points
t_eval = np.linspace(t_start, t_end, n_points)

# Initial state: F with a small perturbation
x0 = F * np.ones(N)
x0[0] += 0.01

solution = solve_ivp(lorenz96, (t_start, t_end), x0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
data = solution.y.T # Shape: (10000, 10)

# ==========================================
# 2. Dataset Preparation
# ==========================================
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

class Lorenz96Dataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size] # (window_size, N)
        y = self.data[idx + self.window_size]       # (N,)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = Lorenz96Dataset(train_data_scaled, WINDOW_SIZE)
test_dataset = Lorenz96Dataset(test_data_scaled, WINDOW_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Model Definition (Residual LSTM)
# ==========================================
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim=N, hidden_dim=HIDDEN_DIM, num_layers=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        last_out = self.layer_norm(last_out)
        
        # Predict delta_x (state increment)
        delta_x = self.fc(last_out) 
        
        # Output = x_last + delta_x
        x_last = x[:, -1, :] 
        predicted_state = x_last + delta_x
        return predicted_state

model = ResidualLSTM(input_dim=N, hidden_dim=HIDDEN_DIM, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 4. Training Loop (with Gradient Clipping)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        
        # Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
        
    train_loss /= len(train_loader.dataset)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{EPOCHS}, Loss: {train_loss:.6f}")

# ==========================================
# 5. Evaluation & Rollout
# ==========================================
model.eval()

# One-step evaluation
all_preds = []
all_true = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x)
        all_preds.append(preds.cpu().numpy())
        all_true.append(batch_y.numpy())
        
preds_unscaled = scaler.inverse_transform(np.concatenate(all_preds, axis=0))
true_unscaled = scaler.inverse_transform(np.concatenate(all_true, axis=0))

rmse = np.sqrt(mean_squared_error(true_unscaled, preds_unscaled))
mae = mean_absolute_error(true_unscaled, preds_unscaled)
r2_mean = np.mean([r2_score(true_unscaled[:, i], preds_unscaled[:, i]) for i in range(N)])

print("\n--- One-step Prediction Metrics ---")
print(f"RMSE:    {rmse:.4f}")
print(f"MAE:     {mae:.4f}")
print(f"R2_mean: {r2_mean:.4f}")

# 500-step Rollout
print("\n--- 500-step Rollout Evaluation ---")
rollout_steps = 500
initial_seq = test_data_scaled[:WINDOW_SIZE] 
true_rollout = test_data[WINDOW_SIZE : WINDOW_SIZE + rollout_steps] 

current_seq = torch.tensor(initial_seq, dtype=torch.float32).unsqueeze(0).to(device)
rollout_preds_scaled = []

with torch.no_grad():
    for _ in range(rollout_steps):
        pred_next = model(current_seq) 
        rollout_preds_scaled.append(pred_next.cpu().numpy()[0])
        pred_next_seq = pred_next.unsqueeze(1) 
        current_seq = torch.cat([current_seq[:, 1:, :], pred_next_seq], dim=1)

rollout_preds_unscaled = scaler.inverse_transform(rollout_preds_scaled)

step_errors = np.linalg.norm(true_rollout - rollout_preds_unscaled, axis=1)
cumulative_rmse = np.sqrt(np.cumsum(step_errors**2) / np.arange(1, rollout_steps + 1))
final_cum_rmse = cumulative_rmse[-1]

print(f"Final Cumulative RMSE (500 steps): {final_cum_rmse:.4f}")

os.makedirs('results', exist_ok=True)
pd.DataFrame({
    'model': 'ResidualLSTM_Optimized',
    'step': np.arange(1, rollout_steps + 1),
    'cumulative_rmse': cumulative_rmse
}).to_csv('results/lorenz96_optimized_lstm_rollout.csv', index=False)
print("Rollout results saved to results/lorenz96_optimized_lstm_rollout.csv")
