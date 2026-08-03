import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import torch
import torch.nn as nn
import numpy as np
from scipy.special import eval_legendre, legendre as legendre_poly

_legendre_deriv_cache = {}

def eval_legendre_derivative(n, x):
    if n not in _legendre_deriv_cache:
        _legendre_deriv_cache[n] = legendre_poly(n).deriv()
    return _legendre_deriv_cache[n](x)

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------
# 1. PHYSICAL & NUMERICAL PARAMETERS
# -----------------------------------------------------------------------------
a = 1.0
L = 1.0

N_x, N_y = 8, 8
N_q_1d = 5
P_deg = 4

# MINI-BATCHING HYPERPARAMETERS (The Speed Fix)
N_t_samples = 10           # Reduced from 50 to 10 (Stochastic time sampling)
N_ic = 500                 # Reduced from 1000
N_b = 100                  # Reduced from 200

def exact_solution(x, y, t):
    return torch.exp(-2 * (torch.pi**2) * a * t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# -----------------------------------------------------------------------------
# 2. PRECOMPUTATION
# -----------------------------------------------------------------------------
roots_1d, weights_1d = np.polynomial.legendre.leggauss(N_q_1d)
xi_g, eta_g = np.meshgrid(roots_1d, roots_1d, indexing='ij')
w_xi, w_eta = np.meshgrid(weights_1d, weights_1d, indexing='ij')

xi_q = xi_g.flatten()
eta_q = eta_g.flatten()
weights_q = (w_xi * w_eta).flatten()
N_q = len(weights_q)

def legendre_diff_mode(m, x):
    return eval_legendre(m + 1, x) - eval_legendre(m - 1, x)

def legendre_diff_mode_deriv(m, x):
    return eval_legendre_derivative(m + 1, x) - eval_legendre_derivative(m - 1, x)

mode_ids = np.arange(1, P_deg + 2)

test_v, test_d_xi, test_d_eta = [], [], []
for m in mode_ids:
    v_m = legendre_diff_mode(m, xi_q)
    dv_m = legendre_diff_mode_deriv(m, xi_q)
    for n in mode_ids:
        v_n = legendre_diff_mode(n, eta_q)
        dv_n = legendre_diff_mode_deriv(n, eta_q)
        test_v.append(v_m * v_n)
        test_d_xi.append(dv_m * v_n)
        test_d_eta.append(v_m * dv_n)

V_test = torch.tensor(np.array(test_v), dtype=torch.float64, device=device)
dV_dxi = torch.tensor(np.array(test_d_xi), dtype=torch.float64, device=device)
dV_deta = torch.tensor(np.array(test_d_eta), dtype=torch.float64, device=device)
N_k = V_test.shape[0]

x_edges = np.linspace(0, 1, N_x + 1)
y_edges = np.linspace(0, 1, N_y + 1)
hx = x_edges[1] - x_edges[0]
hy = y_edges[1] - y_edges[0]
detJ_scalar = 0.25 * hx * hy

dV_dx_ref = (dV_dxi * (2.0 / hx)).cpu().numpy()
dV_dy_ref = (dV_deta * (2.0 / hy)).cpu().numpy()

elem_quad_x, elem_quad_y = [], []
for e_x in range(N_x):
    x1, x2 = x_edges[e_x], x_edges[e_x + 1]
    for e_y in range(N_y):
        y1, y2 = y_edges[e_y], y_edges[e_y + 1]
        x_phys = 0.5 * (x1 + x2) + 0.5 * hx * xi_q
        y_phys = 0.5 * (y1 + y2) + 0.5 * hy * eta_q
        elem_quad_x.append(x_phys)
        elem_quad_y.append(y_phys)

N_e = N_x * N_y

X_quad_spatial = torch.tensor(np.array(elem_quad_x), dtype=torch.float64, device=device)
Y_quad_spatial = torch.tensor(np.array(elem_quad_y), dtype=torch.float64, device=device)
W_detJ = torch.tensor(np.full((N_e, N_q), detJ_scalar) * weights_q, dtype=torch.float64, device=device)

dV_dx = torch.tensor(np.broadcast_to(dV_dx_ref, (N_e, N_k, N_q)).copy(), dtype=torch.float64, device=device)
dV_dy = torch.tensor(np.broadcast_to(dV_dy_ref, (N_e, N_k, N_q)).copy(), dtype=torch.float64, device=device)

# -----------------------------------------------------------------------------
# 3. MODEL SETUP
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
# 4. FAST MINI-BATCHED TRAINING LOOP
# -----------------------------------------------------------------------------
epochs = 15000
print_interval = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    start_event.record()

total_time_ms = 0.0

for i in range(epochs + 1):
    optimizer.zero_grad()

    # Dynamic IC Points
    x_ic = torch.rand(N_ic, 1, dtype=torch.float64, device=device)
    y_ic = torch.rand(N_ic, 1, dtype=torch.float64, device=device)
    t_ic = torch.zeros(N_ic, 1, dtype=torch.float64, device=device)
    X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
    u_ic = exact_solution(x_ic, y_ic, t_ic)

    # Dynamic BC Points
    t_b = torch.rand(N_b, 1, dtype=torch.float64, device=device)
    edge_choice = torch.randint(0, 4, (N_b, 1), device=device)
    x_b = torch.rand(N_b, 1, dtype=torch.float64, device=device)
    y_b = torch.rand(N_b, 1, dtype=torch.float64, device=device)

    x_b = torch.where(edge_choice == 0, torch.zeros_like(x_b), x_b)
    x_b = torch.where(edge_choice == 1, torch.ones_like(x_b), x_b)
    y_b = torch.where(edge_choice == 2, torch.zeros_like(y_b), y_b)
    y_b = torch.where(edge_choice == 3, torch.ones_like(y_b), y_b)
    X_b = torch.cat([x_b, y_b, t_b], dim=1)

    loss_ic = mse(model(X_ic), u_ic)
    loss_bc = mse(model(X_b), torch.zeros_like(X_b[:, 0:1]))

    # Stochastic Temporal Sampling (N_t = 10)
    t_samples = torch.rand(N_t_samples, 1, 1, dtype=torch.float64, device=device)
    X_elem = X_quad_spatial.unsqueeze(0).expand(N_t_samples, -1, -1)
    Y_elem = Y_quad_spatial.unsqueeze(0).expand(N_t_samples, -1, -1)
    T_elem = t_samples.expand(-1, N_e, N_q)

    X_v = torch.stack([X_elem.flatten(), Y_elem.flatten(), T_elem.flatten()], dim=1)
    X_v.requires_grad_(True)

    u_pred = model(X_v)

    grads = torch.autograd.grad(u_pred, X_v, torch.ones_like(u_pred), create_graph=True)[0]
    u_x = grads[:, 0:1].reshape(N_t_samples, N_e, N_q)
    u_y = grads[:, 1:2].reshape(N_t_samples, N_e, N_q)
    u_t = grads[:, 2:3].reshape(N_t_samples, N_e, N_q)

    R_mass = torch.einsum('teq, kq, eq -> tek', u_t, V_test, W_detJ)
    R_stiff_x = torch.einsum('teq, ekq, eq -> tek', u_x, dV_dx, W_detJ)
    R_stiff_y = torch.einsum('teq, ekq, eq -> tek', u_y, dV_dy, W_detJ)

    R_total = R_mass + a * (R_stiff_x + R_stiff_y)
    loss_phys = torch.mean(R_total ** 2)

    loss = loss_ic * 3.0 + loss_bc * 1.0 + loss_phys * 12.0
    loss.backward()
    optimizer.step()

    if i % print_interval == 0:
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            step_time_ms = elapsed_ms / print_interval if i > 0 else elapsed_ms
            total_time_ms += elapsed_ms

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            total_pts = N_t_samples * N_e * N_q + N_ic + N_b
            throughput = (total_pts / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0

            print(f"Iter {i}: Loss={loss.item():.4e} | IC={loss_ic.item():.3e} | BC={loss_bc.item():.3e} | Phys={loss_phys.item():.3e}")
            print(f" ↳ Avg Step Time: {step_time_ms:.3f} ms | Peak VRAM: {peak_vram_mb:.2f} MB | Throughput: {throughput:.0f} pts/sec")
            print("-" * 80)

            start_event.record()
        else:
            print(f"Iter {i}: Loss={loss.item():.4e} | IC={loss_ic.item():.3e} | BC={loss_bc.item():.3e} | Phys={loss_phys.item():.3e}")

        scheduler.step(loss.item())

if torch.cuda.is_available():
    print(f"Total FastVPINN Training Time: {total_time_ms / 1000.0:.2f} seconds")

# --- Export ---

torch.save(model.state_dict(), "fastvpinn_model.pt")
print("Model exported to fastvpinn_model.pt")
print(f"Model successfully saved to fastvpinn_model.pt")