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

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Lorenz-96 Data Generation
# ==========================================
N = 10  # Dimension
F = 8.0 # Forcing constant

def lorenz96(t, x):
    """Lorenz-96 model with periodic boundary conditions"""
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i+1)%N] - x[i-2]) * x[i-1] - x[i] + F
    return dxdt

print("Generating Lorenz-96 data...")
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
WINDOW_SIZE = 20
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 3. Model Definition (Phase 1: Residual LSTM with LayerNorm)
# ==========================================
class ResidualLSTM(nn.Module):
    def __init__(self, input_dim=N, hidden_dim=64, num_layers=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch, window, N)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :] # (batch, hidden_dim)
        last_out = self.layer_norm(last_out)
        
        # Predict delta_x
        delta_x = self.fc(last_out) # (batch, N)
        
        # x_last + delta_x
        x_last = x[:, -1, :] # (batch, N)
        predicted_state = x_last + delta_x
        return predicted_state

model = ResidualLSTM(input_dim=N, hidden_dim=64, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==========================================
# 4. Training Loop (with Gradient Clipping)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Training on {device}...")
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)
        
    train_loss /= len(train_loader.dataset)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.6f}")

# ==========================================
# 5. Evaluation
# ==========================================
model.eval()

# One-step evaluation on test set
all_preds = []
all_true = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x)
        all_preds.append(preds.cpu().numpy())
        all_true.append(batch_y.numpy())
        
preds_scaled = np.concatenate(all_preds, axis=0)
true_scaled = np.concatenate(all_true, axis=0)

preds_unscaled = scaler.inverse_transform(preds_scaled)
true_unscaled = scaler.inverse_transform(true_scaled)

mse = mean_squared_error(true_unscaled, preds_unscaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_unscaled, preds_unscaled)
r2_mean = np.mean([r2_score(true_unscaled[:, i], preds_unscaled[:, i]) for i in range(N)])

print("\n--- One-step Prediction ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2_mean: {r2_mean:.4f}")

# 500-step Rollout
print("\n--- 500-step Rollout Evaluation ---")
rollout_steps = 500
initial_seq = test_data_scaled[:WINDOW_SIZE] # (WINDOW_SIZE, N)
true_rollout = test_data[WINDOW_SIZE : WINDOW_SIZE + rollout_steps] # Unscaled

current_seq = torch.tensor(initial_seq, dtype=torch.float32).unsqueeze(0).to(device) # (1, WINDOW_SIZE, N)
rollout_preds_scaled = []

with torch.no_grad():
    for _ in range(rollout_steps):
        pred_next = model(current_seq) # (1, N)
        rollout_preds_scaled.append(pred_next.cpu().numpy()[0])
        
        # Update sequence: drop oldest, append newest
        pred_next_seq = pred_next.unsqueeze(1) # (1, 1, N)
        current_seq = torch.cat([current_seq[:, 1:, :], pred_next_seq], dim=1)

rollout_preds_unscaled = scaler.inverse_transform(rollout_preds_scaled)

# Calculate Cumulative RMSE State
step_errors = np.linalg.norm(true_rollout - rollout_preds_unscaled, axis=1)
cumulative_rmse = np.sqrt(np.cumsum(step_errors**2) / np.arange(1, rollout_steps + 1))
final_cum_rmse = cumulative_rmse[-1]

print(f"Final Cumulative RMSE (500 steps): {final_cum_rmse:.4f}")

os.makedirs('results', exist_ok=True)
pd.DataFrame({
    'model': 'LSTM_Phase1',
    'step': np.arange(1, rollout_steps + 1),
    'cumulative_rmse': cumulative_rmse
}).to_csv('results/lstm_phase1_rollout.csv', index=False)
print("Rollout results saved to results/lstm_phase1_rollout.csv")
