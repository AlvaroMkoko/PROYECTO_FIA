"""
evaluate.py
-----------
Etapa 3 del pipeline DVC.
Carga el modelo entrenado y los datos de prueba,
calcula las métricas finales y las guarda en metrics/scores.json
"""

import sys
import os
import pickle
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR  = os.path.dirname(__file__)
NN_DIR       = os.path.join(SCRIPTS_DIR, '..')
DATA_PATH    = os.path.join(NN_DIR, 'data', 'processed.pkl')
PARAMS_PATH  = os.path.join(NN_DIR, 'params.yaml')
MODEL_PATH   = os.path.join(NN_DIR, 'models', 'best_model.pth')
METRICS_PATH = os.path.join(NN_DIR, 'metrics', 'scores.json')

# ─── Cargar datos y parámetros ────────────────────────────────────────────────
print("[evaluate] Cargando datos y modelo...")
with open(DATA_PATH, 'rb') as fh:
    d = pickle.load(fh)
X_test, y_test = d['X_test'], d['y_test']
n_features = d['n_features']

with open(PARAMS_PATH, 'r') as fh:
    params = yaml.safe_load(fh)
hidden_size = params['train']['hidden_size']
dropout     = params['train']['dropout']

# ─── Reconstruir arquitectura y cargar pesos ──────────────────────────────────
class ChessNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_size // 2, 2)
        )
    def forward(self, x):
        return self.net(x)

model = ChessNN(n_features, hidden_size, dropout)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# ─── Predicciones ─────────────────────────────────────────────────────────────
with torch.no_grad():
    outputs = model(torch.tensor(X_test, dtype=torch.float32))
    _, preds = torch.max(outputs, 1)
preds = preds.numpy()

# ─── Métricas ─────────────────────────────────────────────────────────────────
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds, average='weighted')

scores = {
    'accuracy': round(float(acc), 4),
    'f1_score': round(float(f1), 4),
}

with open(METRICS_PATH, 'w') as fh:
    json.dump(scores, fh, indent=2)

print("[evaluate] Métricas del modelo final:")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"[evaluate] Guardadas en: {METRICS_PATH}")
print()
print(classification_report(y_test, preds, target_names=['Negras (0)', 'Blancas (1)']))
