from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
torch.set_default_dtype(torch.float64)

# 1. Exact Model Architecture
class FCN(nn.Module):
    def __init__(self, input, output, hidden, layers):
        super().__init__()
        activation = nn.SiLU
        self.pinn = nn.Sequential(
            nn.Linear(input, hidden),
            activation(),
            *[nn.Sequential(nn.Linear(hidden, hidden), activation()) for _ in range(layers)],
            nn.Linear(hidden, output)
        )

    def forward(self, x):
        return self.pinn(x)

# Global variables for model state
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Server Lifespan Configuration (Boot Sequence)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Initialize the raw model and load weights
    raw_model = FCN(input=3, output=1, hidden=64, layers=4)
    raw_model.load_state_dict(torch.load('model.pt', map_location=device))
    raw_model.to(device)
    raw_model.eval()
    
    # Compile the model to a static graph via TorchScript tracing
    example_input = torch.rand(1, 3, dtype=torch.float64).to(device)
    model = torch.jit.trace(raw_model, example_input)
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
    # Convert validated input data to a 2D tensor match training dimensions
    input_tensor = torch.tensor([[data.x, data.y, data.t]], dtype=torch.float64).to(device)
    
    # Run inference without gradient tracking to save compute
    with torch.no_grad():
        prediction = model(input_tensor)
    
    return {"predicted_temperature": float(prediction.item())}

# Mount the frontend directory to serve static assets at the root path
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")