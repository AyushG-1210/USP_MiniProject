import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 1. PHYSICAL & NUMERICAL PARAMETERS
# -----------------------------------------------------------------------------
a = 1.0
L = 1.0

# MATCHING POINT BUDGETS TO FASTVPINN
N_f = 16000                # Matches FastVPINN (10 time slices * 64 elements * 25 quad points)
N_ic = 500
N_b = 100

def exact_solution(x, y, t):
    return torch.exp(-2 * (torch.pi**2) * a * t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# -----------------------------------------------------------------------------
# 2. MODEL SETUP
# -----------------------------------------------------------------------------
class FCN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden=64, layers=4):
        super().__init__()
        activation = nn.SiLU
        self.pinn = nn.Sequential(
            nn.Linear(input_dim, hidden),
            activation(),
            *[nn.Sequential(nn.Linear(hidden, hidden), activation()) for _ in range(layers)],
            nn.Linear(hidden, output_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.pinn(x)

model = FCN().to(device)

# -----------------------------------------------------------------------------
# 3. TRAINING LOOP
# -----------------------------------------------------------------------------
epochs = 15000
print_interval = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

# Scheduler synced to periodic loop (every 1000 iters), so patience is 5 = 5000 iters
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    start_event.record()

total_time_ms = 0.0

for i in range(epochs + 1):
    optimizer.zero_grad()

    # DYNAMIC IC POINTS
    x_ic = torch.rand(N_ic, 1, dtype=torch.float64, device=device)
    y_ic = torch.rand(N_ic, 1, dtype=torch.float64, device=device)
    t_ic = torch.zeros(N_ic, 1, dtype=torch.float64, device=device)
    X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
    u_ic = exact_solution(x_ic, y_ic, t_ic)

    # DYNAMIC BC POINTS
    t_b = torch.rand(N_b, 1, dtype=torch.float64, device=device)
    edge_choice = torch.randint(0, 4, (N_b, 1), device=device)
    x_b = torch.rand(N_b, 1, dtype=torch.float64, device=device)
    y_b = torch.rand(N_b, 1, dtype=torch.float64, device=device)

    x_b = torch.where(edge_choice == 0, torch.zeros_like(x_b), x_b)
    x_b = torch.where(edge_choice == 1, torch.ones_like(x_b), x_b)
    y_b = torch.where(edge_choice == 2, torch.zeros_like(y_b), y_b)
    y_b = torch.where(edge_choice == 3, torch.ones_like(y_b), y_b)
    X_b = torch.cat([x_b, y_b, t_b], dim=1)

    # DYNAMIC COLLOCATION POINTS (Strong Form)
    x_f = torch.rand(N_f, 1, dtype=torch.float64, device=device, requires_grad=True)
    y_f = torch.rand(N_f, 1, dtype=torch.float64, device=device, requires_grad=True)
    t_f = torch.rand(N_f, 1, dtype=torch.float64, device=device, requires_grad=True)
    X_f = torch.cat([x_f, y_f, t_f], dim=1)

    # 1. IC and BC Losses
    loss_ic = mse(model(X_ic), u_ic)
    loss_bc = mse(model(X_b), torch.zeros_like(X_b[:, 0:1]))

    # 2. Strong-Form Physics Loss (Requires 2nd-order derivatives)
    u_pred = model(X_f)

    # First derivatives
    grads = torch.autograd.grad(u_pred, X_f, torch.ones_like(u_pred), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    # Second derivatives (Hessians)
    u_xx = torch.autograd.grad(u_x, X_f, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X_f, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    residual = u_t - a * (u_xx + u_yy)
    loss_phys = mse(residual, torch.zeros_like(residual))

    # 3. Total Loss
    loss = loss_ic * 3.0 + loss_bc * 1.0 + loss_phys * 12.0
    loss.backward()
    optimizer.step()

    # 4. Async Profiling Block
    if i % print_interval == 0:
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            step_time_ms = elapsed_ms / print_interval if i > 0 else elapsed_ms
            total_time_ms += elapsed_ms

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            total_pts = N_f + N_ic + N_b
            throughput = (total_pts / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0

            print(f"Iter {i}: Loss={loss.item():.4e} | IC={loss_ic.item():.3e} | BC={loss_bc.item():.3e} | Phys={loss_phys.item():.3e}")
            print(f" ↳ Avg Step Time: {step_time_ms:.3f} ms | Peak VRAM: {peak_vram_mb:.2f} MB | Throughput: {throughput:.0f} pts/sec")
            print("-" * 80)

            start_event.record() # Restart async timer
        else:
            print(f"Iter {i}: Loss={loss.item():.4e} | IC={loss_ic.item():.3e} | BC={loss_bc.item():.3e} | Phys={loss_phys.item():.3e}")

        # Pull loss to CPU only periodically
        scheduler.step(loss.item())

if torch.cuda.is_available():
    print(f"Total Standard PINN Training Time: {total_time_ms / 1000.0:.2f} seconds")

# --- Export ---
model.eval()
torch.save(model, "model_standard.pt")
print("Model exported to model_standard.pt")