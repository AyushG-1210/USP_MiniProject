from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

torch.set_default_dtype(torch.float64)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = BASE_DIR / "model_fastvpinn.pt"

# 1. Exact Model Architecture
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

# Global variables for model state
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Server Lifespan Configuration (Boot Sequence)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    raw_model = FCN(input_dim=3, output_dim=1, hidden=64, layers=4)
    raw_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    raw_model.to(device)
    raw_model.eval()

    example_input = torch.rand(1, 3, dtype=torch.float64).to(device)
    model = torch.jit.trace(raw_model, example_input)
    model = torch.jit.freeze(model)
    yield

# Initialize FastAPI with lifespan management
app = FastAPI(lifespan=lifespan)

# 3. Input Validation Schema
class CoordinateInput(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Spatial coordinate x in [0, 1]")
    y: float = Field(..., ge=0.0, le=1.0, description="Spatial coordinate y in [0, 1]")
    t: float = Field(..., ge=0.0, le=1.0, description="Temporal coordinate t in [0, 1]")

# 4. Inference Endpoint
@app.post("/predict")
async def predict(data: CoordinateInput):
    if model is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})

    input_tensor = torch.tensor([[data.x, data.y, data.t]], dtype=torch.float64).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    return JSONResponse(content={"predicted_temperature": float(prediction.item())})

# Mount the frontend directory to serve static assets at the root path
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")