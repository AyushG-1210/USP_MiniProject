# PINN-Ops: Low-Latency SciML Inference Engine with GitOps Continuous Validation

![PINN Inference Interface Demo](frontend/Demo.gif)

A production inference service for a **weak-form (variational) Physics-Informed Neural
Network** — a spatio-temporal PDE surrogate trained using a FastVPINN-style tensor-
contraction formulation, served through a TorchScript-compiled, multi-worker FastAPI
stack with CI-gated performance regression testing.

This isn't just a model wrapped in an API — the underlying network is trained via the
weak (variational) form of the 2D heat equation, using precomputed Legendre-polynomial
test functions and Gauss-Legendre quadrature rather than raw pointwise collocation, so
the model itself embeds a nontrivial piece of numerical PDE methodology, not just a
generic regression fit.

---

## Project Status

Deployed and benchmarked. CI pipeline active on `main`.

---

## Why this exists

Classical numerical PDE solvers (finite element, finite difference) are accurate but
slow to query at inference time — every new coordinate requires re-solving or
interpolating from a stored grid. A trained neural surrogate, once fit to the governing
PDE, collapses that cost to a single forward pass. This project builds and ships one
end-to-end: from the PDE formulation, through training-time architectural choices
(weak-form loss, quadrature-based integration), to a hardened, load-tested production
API.

---

## Architecture

```
[ Spatial/Temporal Inputs ] -> (x, y, t)
            |
            v
+-----------------+
|  Frontend (UI)  |  <- Served via FastAPI StaticFiles
+-----------------+
            |
            v  (Async JSON)
+------------------------------------------------------------+
| Docker Container  (non-root: appuser / Alpine Linux)       |
|                                                            |
|  [ Port 8000 ]                                             |
|       |                                                    |
|       v                                                    |
|  Master Uvicorn Process (load balancer)                    |
|    |          |          |          |                      |
|    v          v          v          v                      |
|  Worker #1  Worker #2  Worker #3  Worker #4                |
|    |          |          |          |                      |
|    +----------+----+-----+----------+                      |
|                    |                                       |
|                    v                                       |
|         Pydantic Input Validation                          |
|                    |                                       |
|                    v                                       |
|         JIT TorchScript Engine (compiled graph)            |
|                    |                                       |
|                    v                                       |
|            [ Tensor Forward Pass ]                         |
+------------------------------------------------------------+
                    |
                    v
        {"predicted_temperature": float}
```

### 1. The model: weak-form PINN

The served model was trained by minimizing the **variational (weak) form** of the 2D heat equation — integrating the PDE residual against precomputed Legendre-difference test functions over a meshed domain, via Gauss-Legendre quadrature, rather than enforcing the PDE pointwise via second-order automatic differentiation. This trades a strong-form PINN's expensive Hessian computation for a single first-order pass plus a batched tensor contraction (`einsum`) against precomputed constants — cheaper to train, and in this project's testing, meaningfully more accurate at matched training budgets.

#### Ablation Study: Strong-Form PINN vs. Weak-Form FastVPINN

To evaluate the impact of the variational weak-form formulation, a controlled 1:1 ablation study was conducted against a standard strong-form PINN baseline using identical network topology (4 hidden layers × 64 units, SiLU activation), double precision (FP64), loss weight ratios ($\lambda_{ic}=3.0, \lambda_{bc}=1.0, \lambda_{phys}=12.0$), and equal point budgets ($N_f = 16,000$).

| Metric / Parameter | Standard PINN (Strong-Form) | FastVPINN (Weak-Form) | Experimental Delta |
| :--- | :---: | :---: | :---: |
| **PDE Loss Formulation** | Pointwise $u_t - a(u_{xx} + u_{yy})$ | Variational integral $\int u_t v + a \nabla u \cdot \nabla v$ | Shipped spatial derivatives to test functions |
| **Autograd Requirement** | 2nd-order spatial Hessians | 1st-order gradients only | Bypassed computational graph unrolling |
| **Total Training Time** | 1,979.78 s | **542.40 s** | **~3.6x faster training** |
| **Relative $L_2$ Error** | 39.64% | **3.09%** | **~12.8x higher accuracy** |
| **Max Absolute Error** | 0.1333 | **0.0152** | **~8.7x lower peak error** |
| **Peak VRAM Allocation** | 1,168.75 MB | **400.51 MB** | **~65% memory reduction** |

#### Key Ablation Insights
* **Autograd Graph Bottleneck:** Strong-form PINNs require computing continuous 2nd-order automatic differentiation ($u_{xx}, u_{yy}$) across 16,000 collocation points per epoch, heavily inflating GPU memory usage and step latency.
* **Loss Landscape Stiffness:** Second-order autograd amplifies high-frequency noise in neural networks, causing the strong-form Adam optimizer to stall at ~39.6% error. FastVPINN's weak formulation uses integration by parts to smooth the loss landscape, achieving 3.09% relative $L_2$ error under identical training budgets.

### 2. Physical dimensional conversion

While the core neural network operates over non-dimensional normalized domain bounds $(x, y, t) \in [0,1]^3$, the frontend client layer dynamically converts dimensionless model outputs into physical engineering quantities:
* **Spatial Scaling**: $x_{\text{real}} = L \cdot x$, $y_{\text{real}} = L \cdot y$
* **Temporal Scaling**: $t_{\text{real}} = \tau \cdot t$ where $\tau = \frac{L^2}{\alpha}$ (using material thermal diffusivity $\alpha$)
* **Temperature Scaling**: $u_{\text{real}} = u_{\text{ref}} + \Delta T \cdot u$

### 3. Compiled inference core

At startup, the trained model is loaded and trace-compiled via TorchScript JIT (`torch.jit.trace` + `torch.jit.freeze`) against a dummy coordinate tensor. This produces a static, serialized computation graph: once compiled, the forward pass no longer re-interprets Python-level model code on every call, reducing per-request overhead compared to eager-mode execution.

### 4. Async serving layer & batching

FastAPI on Uvicorn's ASGI server, running four parallel worker processes to use available CPU cores concurrently. Includes dedicated `/predict-batch` endpoints for vectorizing multi-point time sweeps in a single forward pass, eliminating HTTP waterfall delays. Incoming JSON payloads are strictly validated by Pydantic schemas (`ge=0.0, le=1.0`).

### 5. Hardened container

Runs on an Alpine Linux base image. A dedicated `appuser` group/user is created at build time, with root dropped entirely before the Uvicorn runtime starts and build tooling stripped from the final image.

### 6. GitOps continuous validation

Performance is a first-class regression test. Every push to `main` triggers a GitHub Actions run that:
1. Builds a clean container image from scratch
2. Launches the container and waits for JIT warmup across all workers
3. Runs `load_server.py` — a multi-threaded stress test tracking full latency distributions
4. Compares p95/p99 tail latency and throughput against fixed quality gates
5. Fails the build and blocks the merge if any gate is breached

---

## Performance Benchmarks

| Metric | Result | Gate | Status |
| :--- | :---: | :---: | :---: |
| Sustained Throughput | **829.23 req/s** | > 100 req/s | ✅ Passed |
| p95 Tail Latency | **16.57 ms** | < 100 ms | ✅ Passed |
| p99 Tail Latency | **19.26 ms** | < 150 ms | ✅ Passed |
| Transaction Success Rate | **100.0%** | 100.0% | ✅ Passed |

These are end-to-end HTTP round-trip numbers (network + Pydantic validation + JIT inference) measured under concurrent multi-threaded stress testing.

---

## Project Structure

```
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI performance validation pipeline
├── PINN/
│   └── fastvpinn_2d_heat.py          # CI performance validation pipeline
├── frontend/
│   ├── index.html                    # UI layout
│   ├── app.js                        # Single-point prediction: fetch + render logic
│   ├── plotter.js                    # Batch prediction + Chart.js trend visualization
│   ├── style.css                     # Styling
│   └── Demo.gif                      # Interface demo
├── server.py                         # FastAPI app — validation, JIT inference, static serving
├── load_server.py                    # Parallel stress test and tail latency reporter
├── Dockerfile                        # Non-root Alpine multi-worker image
├── model_fastvpinn.pt                # Serialized model weights (see Model Weights below)
├── FastVPINN_Project_Reference.md    # Full derivation + debugging journal for the model
└── requirements.txt                  # Version-locked Python dependencies
```

---

## Prerequisites

- Docker (tested on 24.x)
- Python 3.12+ (only needed to run `load_server.py` outside Docker)
- `model_fastvpinn.pt` — serialized weights must be present in the repo root before
  building. See [Model Weights](#model-weights) below.

---

## Quickstart

**1. Build the image**
```bash
docker build -t pinn-inference:latest .
```

**2. Run the container**
```bash
docker run -d -p 8000:8000 --name pinn-container pinn-inference:latest
```

**3. Open the UI**
```
http://localhost:8000
```

**4. Run the load test manually**
```bash
python load_server.py
```

Fires the same stress test the CI pipeline runs and prints a full latency distribution
report.

---

## Model Weights

`model_fastvpinn.pt` holds the trained weights for a PINN surrogate predicting
spatiotemporal temperature fields `u(x, y, t)` over the unit domain `[0,1]³`, trained
via the weak-form FastVPINN methodology described in


**If you retrain or swap in a different model:** make sure the saved artifact matches
what `server.py` expects — a `state_dict` from `torch.save(model.state_dict(), path)`,
not a full pickled model or a pre-traced `torch.jit` module (those load differently and
will fail against the current loading code).

---

## CI Pipeline Details

The workflow in `.github/workflows/ci.yml` runs on every push to `main`:

```yaml
# Simplified outline — see ci.yml for full config
- Build Docker image
- Start container (docker run -d ...)
- Sleep N seconds   # absorb JIT warmup across all 4 workers
- Run load_server.py
- Assert: throughput > 100 req/s
- Assert: p95 < 100ms, p99 < 150ms
- Assert: success rate == 100%
```

Threshold constants live at the top of `load_server.py`; the trigger branch is set in
`ci.yml`'s `on: push: branches:` field.

---
