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

# ==========================================
# 2. Dataset Preparation (Multi-step)
# ==========================================
WINDOW_SIZE = 20
M_STEPS = 5 # Multi-step unroll
split_idx = int(len(data) * 0.8)

train_data = data[:split_idx]
test_data = data[split_idx:]

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

class Lorenz96MultiStepDataset(Dataset):
    def __init__(self, data, window_size, m_steps):
        self.data = data
        self.window_size = window_size
        self.m_steps = m_steps
        
    def __len__(self):
        return len(self.data) - self.window_size - self.m_steps + 1
        
    def __getitem__(self, idx):
        # Input sequence: window_size
        x = self.data[idx : idx + self.window_size] 
        # Future targets: m_steps
        y = self.data[idx + self.window_size : idx + self.window_size + self.m_steps]
        # Also need the "true" inputs for teacher forcing for the next m_steps-1
        # The true input at step k is self.data[idx + k : idx + window_size + k]
        # We can just return a large chunk and slice in the loop
        full_chunk = self.data[idx : idx + self.window_size + self.m_steps]
        return torch.tensor(full_chunk, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

train_dataset = Lorenz96MultiStepDataset(train_data_scaled, WINDOW_SIZE, M_STEPS)
test_dataset = Lorenz96MultiStepDataset(test_data_scaled, WINDOW_SIZE, 1) # Test one step for metrics

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==========================================
# 3. Model Definition
# ==========================================
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

model = ResidualLSTM(input_dim=N, hidden_dim=64, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==========================================
# 4. Training Loop (Multi-step + Scheduled Sampling)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Training on {device} with Multi-step (M={M_STEPS}) & Scheduled Sampling...")
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    # Scheduled sampling probability (1.0 -> 0.0)
    p_teacher_forcing = max(0.0, 1.0 - (epoch / (EPOCHS * 0.8))) 
    
    for full_chunk, batch_y in train_loader:
        full_chunk, batch_y = full_chunk.to(device), batch_y.to(device)
        batch_size = full_chunk.size(0)
        
        optimizer.zero_grad()
        loss = 0.0
        
        # Initial window
        current_seq = full_chunk[:, :WINDOW_SIZE, :]
        
        for m in range(M_STEPS):
            pred_next = model(current_seq)
            loss += criterion(pred_next, batch_y[:, m, :])
            
            if m < M_STEPS - 1:
                # Decide whether to use true data or prediction for the next step
                use_true = np.random.rand() < p_teacher_forcing
                
                if use_true:
                    current_seq = full_chunk[:, m+1 : WINDOW_SIZE+m+1, :]
                else:
                    pred_next_seq = pred_next.unsqueeze(1).detach() # Detach to avoid BPTT through predictions? Actually keeping gradients might help, but detach is standard scheduled sampling. Let's keep gradients for multi-step loss!
                    # Wait, if we keep gradients, it's fully differentiable rollout (which is good).
                    pred_next_seq = pred_next.unsqueeze(1)
                    current_seq = torch.cat([current_seq[:, 1:, :], pred_next_seq], dim=1)
                    
        loss /= M_STEPS
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * batch_size
        
    train_loss /= len(train_loader.dataset)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.6f}, P_TF: {p_teacher_forcing:.2f}")

# ==========================================
# 5. Evaluation
# ==========================================
model.eval()

# One-step evaluation on test set
all_preds = []
all_true = []
with torch.no_grad():
    for full_chunk, batch_y in test_loader:
        batch_x = full_chunk[:, :WINDOW_SIZE, :].to(device)
        preds = model(batch_x)
        all_preds.append(preds.cpu().numpy())
        all_true.append(batch_y[:, 0, :].numpy())
        
preds_scaled = np.concatenate(all_preds, axis=0)
true_scaled = np.concatenate(all_true, axis=0)

preds_unscaled = scaler.inverse_transform(preds_scaled)
true_unscaled = scaler.inverse_transform(true_scaled)

rmse = np.sqrt(mean_squared_error(true_unscaled, preds_unscaled))
mae = mean_absolute_error(true_unscaled, preds_unscaled)
r2_mean = np.mean([r2_score(true_unscaled[:, i], preds_unscaled[:, i]) for i in range(N)])

print("\n--- One-step Prediction ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
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

pd.DataFrame({
    'model': 'LSTM_Phase2',
    'step': np.arange(1, rollout_steps + 1),
    'cumulative_rmse': cumulative_rmse
}).to_csv('results/lstm_phase2_rollout.csv', index=False)
print("Rollout results saved to results/lstm_phase2_rollout.csv")
