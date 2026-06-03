import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Lorenz-96 Data Generation
# ==========================================
N = 10
F = 8.0

def lorenz96(t, x):
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i+1)%N] - x[i-2]) * x[i-1] - x[i] + F
    return dxdt

t_start, t_end = 0, 100
n_points = 10000
t_eval = np.linspace(t_start, t_end, n_points)

x0 = F * np.ones(N)
x0[0] += 0.01

solution = solve_ivp(lorenz96, (t_start, t_end), x0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
data = solution.y.T

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
        x = self.data[idx : idx + self.window_size] 
        y = self.data[idx + self.window_size]       
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ResidualLSTM(nn.Module):
    def __init__(self, input_dim=N, hidden_dim=64, num_layers=1):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        last_out = self.layer_norm(last_out)
        delta_x = self.fc(last_out)
        x_last = x[:, -1, :]
        return x_last + delta_x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 20

windows_to_test = [10, 20, 50]
results = []
os.makedirs('results', exist_ok=True)

for w in windows_to_test:
    print(f"\n==========================================")
    print(f"Testing Window Size = {w}")
    print(f"==========================================")
    
    train_dataset = Lorenz96Dataset(train_data_scaled, w)
    test_dataset = Lorenz96Dataset(test_data_scaled, w)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = ResidualLSTM(input_dim=N, hidden_dim=64, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    # Evaluation
    model.eval()
    
    # 500-step Rollout
    rollout_steps = 500
    initial_seq = test_data_scaled[:w]
    true_rollout = test_data[w : w + rollout_steps]
    
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
    
    print(f"Window {w} - Final Cumulative RMSE (500 steps): {final_cum_rmse:.4f}")
    
    pd.DataFrame({
        'model': f'LSTM_Phase3_W{w}',
        'step': np.arange(1, rollout_steps + 1),
        'cumulative_rmse': cumulative_rmse
    }).to_csv(f'results/lstm_phase3_w{w}_rollout.csv', index=False)
    
    results.append({'window_size': w, 'final_rollout_rmse': final_cum_rmse})

pd.DataFrame(results).to_csv('results/lstm_phase3_summary.csv', index=False)
print("\nPhase 3 summary saved.")
