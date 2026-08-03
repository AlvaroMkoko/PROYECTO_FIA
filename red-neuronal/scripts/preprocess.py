"""
preprocess.py
-------------
Etapa 1 del pipeline DVC.
Carga games.csv, aplica el mismo preprocesamiento que Funciones.py
y guarda el resultado serializado en data/processed.pkl
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

# Añadir el directorio raíz al path para importar Funciones.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import Funciones as f

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Reproducibilidad ────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── Carga y preprocesamiento ─────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'games.csv')
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'chess_random_forest_model.joblib')

print("[preprocess] Cargando datos...")
data = f.get_data(DATA_PATH)
data = f.filter(data, 'increment_code', 0.02)
data = f.filter(data, 'opening_eco', 0.02)
data = f.code(data)
data = f.balance(data)

# Recuperar las características seleccionadas por RFE
rf_data = joblib.load(RF_MODEL_PATH)
list_features = list(rf_data['feature_names'])

X = data[list_features]
y = data['winner']

# Escalar con StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
)

# Guardar todo en un único pickle
output = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train.values,
    'y_test': y_test.values,
    'feature_names': list_features,
    'n_features': X_train.shape[1],
}

output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed.pkl')
with open(output_path, 'wb') as fh:
    pickle.dump(output, fh)

print(f"[preprocess] OK — {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} de prueba.")
print(f"[preprocess] Guardado en: {output_path}")
