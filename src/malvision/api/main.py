"""
FastAPI inference service for MalVisionNet.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from pathlib import Path
import onnxruntime as ort
import yaml

from ..representations import GrayscaleConverter, RGBTrigramConverter
from ..logger import logger

app = FastAPI(title="MalVisionNet Inference API")

# Load model configuration
CONFIG_PATH = Path("configs/train.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Initialize ONNX runtime
model_path = Path(config['final_model_path']).with_suffix('.onnx')
if model_path.exists():
    sess = ort.InferenceSession(str(model_path))
    logger.info("Model loaded successfully")
else:
    sess = None
    logger.warning("Model file not found!")

if config['representation'] == 'grayscale':
    converter = GrayscaleConverter()
else:
    converter = RGBTrigramConverter()

class PredictionResponse(BaseModel):
    label: str
    probability: float
    sha256: str
    metadata: dict

@app.get("/health")
def health():
    return {"status": "ready" if sess else "model_not_loaded"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not sess:
        raise HTTPException(status_code=503, detail="Model not loaded")

    bytes_data = await file.read()

    # Convert and predict
    image, metadata = converter.convert(bytes_data)

    # Prepare input for ONNX
    input_tensor = image[np.newaxis, np.newaxis, ...] if config['representation'] == 'grayscale' else image[np.newaxis, ...]
    if config['representation'] == 'rgb':
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

    # Inference
    input_name = sess.get_inputs()[0].name
    prob = sess.run(None, {input_name: input_tensor})[0]
    prob = float(1 / (1 + np.exp(-prob)))

    return PredictionResponse(
        label="malicious" if prob > 0.5 else "benign",
        probability=prob,
        sha256=metadata.get('sha256', 'unknown'),
        metadata=metadata
    )
