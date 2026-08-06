from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import json
import csv
import os
from datetime import datetime

from monitoring import generate_report as _generate_report

app = FastAPI(
    title="Chess Game Predictor",
    description="Predice el ganador de una partida de ajedrez usando una red neuronal entrenada sobre datos de Lichess.",
    version="1.0.0"
)

# ── Arquitectura (debe ser idéntica a la del notebook) ────────────────────────
class ChessNN_GS(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super(ChessNN_GS, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x):
        return self.net(x)

# ── Cargar artefactos al iniciar ──────────────────────────────────────────────
ARTIFACTS = "artifacts"

with open(f"{ARTIFACTS}/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

scaler = joblib.load(f"{ARTIFACTS}/scaler.joblib")

model = ChessNN_GS(input_size=len(FEATURE_NAMES), hidden_size=128, dropout=0.3)
model.load_state_dict(torch.load(f"{ARTIFACTS}/chess_nn_best.pt", map_location="cpu"))
model.eval()

# ── Log de predicciones ───────────────────────────────────────────────────────
LOG_FILE = "logs/predictions.csv"
os.makedirs("logs", exist_ok=True)

def log_prediction(features: dict, prediction: str, confidence: float):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "prediction", "confidence"] + FEATURE_NAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            **features
        })

# ── Schema de entrada ─────────────────────────────────────────────────────────
class ChessInput(BaseModel):
    features: dict[str, float]

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "white_rating":          0.65,
                    "black_rating":          0.40,
                    "opening_ply":           0.30,
                    "victory_status_mate":   0.0,
                    "victory_status_resign": 1.0,
                    "opening_eco_B00":       0.0,
                    "opening_eco_B01":       0.0,
                    "opening_eco_C00":       1.0,
                    "opening_eco_C41":       0.0,
                    "opening_eco_D00":       0.0,
                    "increment_code_10+0":   1.0,
                    "increment_code_15+0":   0.0
                }
            }
        }
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Chess Game Predictor",
        "version": "1.0.0",
        "features_expected": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(data: ChessInput):
    # Validar que lleguen todas las features
    missing = [f for f in FEATURE_NAMES if f not in data.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Features faltantes: {missing}"
        )

    # Construir el vector en el orden correcto
    vector = np.array([[data.features[f] for f in FEATURE_NAMES]])

    # Escalar
    vector_scaled = scaler.transform(vector)
    tensor = torch.tensor(vector_scaled, dtype=torch.float32)

    # Predecir
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    prediction = "white" if pred_idx == 1 else "black"

    # Loggear
    log_prediction(data.features, prediction, confidence)

    return {
        "winner": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {
            "black": round(probs[0][0].item(), 4),
            "white": round(probs[0][1].item(), 4)
        }
    }

@app.get("/example")
def example():
    """Retorna un ejemplo de request realista para POST /predict"""

    example_features = {f: 0.0 for f in FEATURE_NAMES}

    # Valores representativos de una partida típica
    # Ratings normalizados: jugador blanco ligeramente más fuerte
    if "white_rating"          in example_features: example_features["white_rating"]          = 0.65
    if "black_rating"          in example_features: example_features["black_rating"]          = 0.40
    if "opening_ply"           in example_features: example_features["opening_ply"]           = 0.30

    # La partida termina en rendición (resign = 1, el resto = 0)
    if "victory_status_resign" in example_features: example_features["victory_status_resign"] = 1.0

    # Apertura francesa (C00 = 1, el resto = 0)
    if "opening_eco_C00"       in example_features: example_features["opening_eco_C00"]       = 1.0

    # Control de tiempo: 10 minutos sin incremento
    if "increment_code_10+0"   in example_features: example_features["increment_code_10+0"]   = 1.0

    return {
        "description": "Copia el valor de 'request' y pégalo en el body de POST /predict",
        "notas": {
            "ratings": "Valores entre 0.0 (mínimo del dataset) y 1.0 (máximo). ~0.41 es rating promedio (~1500 ELO)",
            "victory_status": "Solo uno puede ser 1 a la vez. Ambos en 0 = outoftime",
            "opening_eco": "Solo uno puede ser 1 a la vez. Todos en 0 = apertura poco frecuente",
            "increment_code": "Solo uno puede ser 1 a la vez. Ambos en 0 = otro control de tiempo"
        },
        "total_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "request": {
            "features": example_features
        }
    }

@app.post("/monitoring/report")
def monitoring_report():
    """Genera un reporte de drift comparando logs vs datos de entrenamiento"""
    try:
        path = _generate_report()
        return {
            "status": "ok",
            "report": path,
            "message": "Reporte generado. Extráelo del contenedor con docker cp."
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))