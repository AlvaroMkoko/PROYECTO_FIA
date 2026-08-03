"""
train_best.py
-------------
Etapa 2 del pipeline DVC.
Lee los datos preprocesados de data/processed.pkl,
lee los hiperparámetros de params.yaml y entrena el modelo
final, guardándolo en models/best_model.pth
"""

import sys
import os
import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(__file__)
NN_DIR      = os.path.join(SCRIPTS_DIR, '..')
DATA_PATH   = os.path.join(NN_DIR, 'data', 'processed.pkl')
PARAMS_PATH = os.path.join(NN_DIR, 'params.yaml')
MODEL_PATH  = os.path.join(NN_DIR, 'models', 'best_model.pth')

# ─── Cargar datos ─────────────────────────────────────────────────────────────
print("[train] Cargando datos preprocesados...")
with open(DATA_PATH, 'rb') as fh:
    d = pickle.load(fh)

X_train, y_train = d['X_train'], d['y_train']
X_test,  y_test  = d['X_test'],  d['y_test']
n_features = d['n_features']

# ─── Cargar hiperparámetros ───────────────────────────────────────────────────
print("[train] Leyendo params.yaml...")
with open(PARAMS_PATH, 'r') as fh:
    params = yaml.safe_load(fh)

lr           = params['train']['lr']
dropout      = params['train']['dropout']
hidden_size  = params['train']['hidden_size']
weight_decay = params['train']['weight_decay']
epochs       = params['train']['epochs']
batch_size   = params['train']['batch_size']

print(f"[train] Hiperparámetros: lr={lr}, dropout={dropout}, "
      f"hidden={hidden_size}, wd={weight_decay}, epochs={epochs}")

# ─── Modelo ───────────────────────────────────────────────────────────────────
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

# ─── DataLoaders ──────────────────────────────────────────────────────────────
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model     = ChessNN(n_features, hidden_size, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# ─── Bucle de entrenamiento ───────────────────────────────────────────────────
print("[train] Entrenando...")
model.train()
for epoch in range(epochs):
    total_loss, correct, total = 0.0, 0, 0
    for bX, by in train_loader:
        optimizer.zero_grad()
        out  = model(bX)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bX.size(0)
        _, preds = torch.max(out, 1)
        correct  += (preds == by).sum().item()
        total    += by.size(0)
    if (epoch + 1) % 10 == 0:
        print(f"  Época {epoch+1}/{epochs} — loss: {total_loss/total:.4f} "
              f"acc: {correct/total:.4f}")

# ─── Guardar modelo ───────────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_PATH)
print(f"[train] Modelo guardado en: {MODEL_PATH}")
