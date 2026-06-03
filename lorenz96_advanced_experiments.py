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
import math

# ==========================================
# Configurations
# ==========================================
torch.manual_seed(42)
np.random.seed(42)

N = 10
F_TRUE = 8.0
F_IMPERFECT = 8.5
WINDOW_SIZE = 20
EPOCHS = 20
BATCH_SIZE = 64
HIDDEN_DIM = 64
LR = 1e-3
M_STEPS = 5 # For scheduled sampling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('results/advanced', exist_ok=True)

# ==========================================
# 1. Data Generation & Physics Functions
# ==========================================
def lorenz96_np(t, x, F=F_TRUE):
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i+1)%N] - x[i-2]) * x[i-1] - x[i] + F
    return dxdt

def lorenz96_tensor(x, F=F_TRUE):
    # x shape: (batch, N)
    dxdt = torch.zeros_like(x)
    for i in range(N):
        dxdt[:, i] = (x[:, (i+1)%N] - x[:, i-2]) * x[:, i-1] - x[:, i] + F
    return dxdt

def rk4_step_tensor(x, dt, F=F_TRUE):
    k1 = lorenz96_tensor(x, F)
    k2 = lorenz96_tensor(x + 0.5 * dt * k1, F)
    k3 = lorenz96_tensor(x + 0.5 * dt * k2, F)
    k4 = lorenz96_tensor(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

t_start, t_end = 0, 100
n_points = 10000
dt = (t_end - t_start) / n_points
t_eval = np.linspace(t_start, t_end, n_points)

x0 = F_TRUE * np.ones(N)
x0[0] += 0.01

print("Generating Lorenz-96 Data...")
solution = solve_ivp(lambda t, x: lorenz96_np(t, x, F_TRUE), (t_start, t_end), x0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
data = solution.y.T

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Since physics constraints and hybrid models mix raw physics and scaled ML, 
# we need tensor scalers for the loss functions.
mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)

def unscale(x_scaled):
    return x_scaled * scale_tensor + mean_tensor

def scale(x_unscaled):
    return (x_unscaled - mean_tensor) / scale_tensor

# ==========================================
# 2. Datasets
# ==========================================
class MultiStepDataset(Dataset):
    def __init__(self, data, window, steps):
        self.data = data
        self.window = window
        self.steps = steps
        
    def __len__(self):
        return len(self.data) - self.window - self.steps + 1
        
    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.window + self.steps]
        return torch.tensor(chunk, dtype=torch.float32)

train_loader = DataLoader(MultiStepDataset(train_data_scaled, WINDOW_SIZE, M_STEPS), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(MultiStepDataset(test_data_scaled, WINDOW_SIZE, 1), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. Model Architectures
# ==========================================
class ResidualLSTM(nn.Module):
    def __init__(self, use_hybrid=False):
        super().__init__()
        self.use_hybrid = use_hybrid
        self.lstm = nn.LSTM(N, HIDDEN_DIM, 1, batch_first=True)
        self.ln = nn.LayerNorm(HIDDEN_DIM)
        self.fc = nn.Linear(HIDDEN_DIM, N)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = self.ln(out[:, -1, :])
        residual = self.fc(last_out)
        
        x_last_scaled = x[:, -1, :]
        if self.use_hybrid:
            # Predict residual on top of imperfect physics
            x_last_unscaled = unscale(x_last_scaled)
            phys_next_unscaled = rk4_step_tensor(x_last_unscaled, dt, F=F_IMPERFECT)
            phys_next_scaled = scale(phys_next_unscaled)
            return phys_next_scaled + residual
        else:
            return x_last_scaled + residual

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, use_hybrid=False):
        super().__init__()
        self.use_hybrid = use_hybrid
        self.embedding = nn.Linear(N, HIDDEN_DIM)
        self.pos_encoder = PositionalEncoding(HIDDEN_DIM)
        encoder_layers = nn.TransformerEncoderLayer(HIDDEN_DIM, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.fc = nn.Linear(HIDDEN_DIM, N)
        
    def forward(self, x):
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        out = self.transformer(emb)
        residual = self.fc(out[:, -1, :])
        
        x_last_scaled = x[:, -1, :]
        if self.use_hybrid:
            x_last_unscaled = unscale(x_last_scaled)
            phys_next_unscaled = rk4_step_tensor(x_last_unscaled, dt, F=F_IMPERFECT)
            phys_next_scaled = scale(phys_next_unscaled)
            return phys_next_scaled + residual
        else:
            return x_last_scaled + residual

# ==========================================
# 4. Training Routines
# ==========================================
def train_model(name, model, use_pinn=False, use_refined_ss=False):
    print(f"\n--- Training {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        # Inverse Sigmoid Decay for Refined SS
        k = 10.0
        if use_refined_ss:
            p_tf = k / (k + math.exp(epoch / (EPOCHS/10)))
        else:
            p_tf = 1.0 # Standard one-step training (always use true past)
            
        for chunk in train_loader:
            chunk = chunk.to(device)
            optimizer.zero_grad()
            loss = 0.0
            
            curr_seq = chunk[:, :WINDOW_SIZE, :]
            
            # Unroll steps
            steps_to_unroll = M_STEPS if use_refined_ss else 1
            
            for m in range(steps_to_unroll):
                pred_next = model(curr_seq)
                true_next = chunk[:, WINDOW_SIZE+m, :]
                
                step_loss = mse_loss(pred_next, true_next)
                
                # PINN Loss
                if use_pinn:
                    pred_unscaled = unscale(pred_next)
                    prev_unscaled = unscale(curr_seq[:, -1, :])
                    # Finite difference approximation of derivative
                    df_dt_approx = (pred_unscaled - prev_unscaled) / dt
                    df_dt_physics = lorenz96_tensor(prev_unscaled, F=F_TRUE)
                    pinn_loss = mse_loss(df_dt_approx, df_dt_physics)
                    step_loss += 0.1 * pinn_loss # Lambda = 0.1
                
                loss += step_loss
                
                if m < steps_to_unroll - 1:
                    use_true = np.random.rand() < p_tf
                    if use_true:
                        curr_seq = chunk[:, m+1 : WINDOW_SIZE+m+1, :]
                    else:
                        pred_next_seq = pred_next.unsqueeze(1)
                        curr_seq = torch.cat([curr_seq[:, 1:, :], pred_next_seq], dim=1)
            
            loss /= steps_to_unroll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.6f} | P_TF: {p_tf:.2f}")
    return model

# ==========================================
# 5. Execution & Rollout
# ==========================================
models = {
    "Baseline_LSTM": train_model("Baseline_LSTM", ResidualLSTM(use_hybrid=False).to(device), use_pinn=False, use_refined_ss=False),
    "PINN_LSTM": train_model("PINN_LSTM", ResidualLSTM(use_hybrid=False).to(device), use_pinn=True, use_refined_ss=False),
    "Transformer": train_model("Transformer", TransformerModel(use_hybrid=False).to(device), use_pinn=False, use_refined_ss=False),
    "Refined_SS_LSTM": train_model("Refined_SS_LSTM", ResidualLSTM(use_hybrid=False).to(device), use_pinn=False, use_refined_ss=True),
    "Hybrid_LSTM": train_model("Hybrid_LSTM", ResidualLSTM(use_hybrid=True).to(device), use_pinn=False, use_refined_ss=False),
    "Ultimate_Hybrid": train_model("Ultimate_Hybrid", TransformerModel(use_hybrid=True).to(device), use_pinn=True, use_refined_ss=True)
}

print("\n--- Running 500-step Rollouts ---")
rollout_steps = 500
true_rollout = test_data[WINDOW_SIZE : WINDOW_SIZE + rollout_steps]
initial_seq = test_data_scaled[:WINDOW_SIZE]

results = []

for name, model in models.items():
    model.eval()
    curr_seq = torch.tensor(initial_seq, dtype=torch.float32).unsqueeze(0).to(device)
    preds_scaled = []
    
    with torch.no_grad():
        for _ in range(rollout_steps):
            pred = model(curr_seq)
            preds_scaled.append(pred.cpu().numpy()[0])
            curr_seq = torch.cat([curr_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)
            
    preds_unscaled = scaler.inverse_transform(preds_scaled)
    step_errors = np.linalg.norm(true_rollout - preds_unscaled, axis=1)
    cum_rmse = np.sqrt(np.cumsum(step_errors**2) / np.arange(1, rollout_steps + 1))
    
    print(f"{name:20s} Final Cum RMSE: {cum_rmse[-1]:.4f}")
    
    pd.DataFrame({'step': np.arange(1, rollout_steps + 1), 'cumulative_rmse': cum_rmse}).to_csv(f'results/advanced/{name}_rollout.csv', index=False)
    results.append({'Model': name, 'Final_Rollout_RMSE': cum_rmse[-1]})

pd.DataFrame(results).to_csv('results/advanced/summary.csv', index=False)
print("\nAll advanced experiments completed and saved in results/advanced/")
