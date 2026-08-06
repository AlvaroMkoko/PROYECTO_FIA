import pandas as pd
import numpy as np
import json
import joblib
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ClassificationQualityMetric
from datetime import datetime

# ── Cargar referencias ────────────────────────────────────────────────────────
ARTIFACTS = "artifacts"
LOGS      = "logs/predictions.csv"
REPORTS   = "reports"

os.makedirs(REPORTS, exist_ok=True)

with open(f"{ARTIFACTS}/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

# ── Cargar datos de entrenamiento como referencia ─────────────────────────────
def load_reference_data():
    """
    Carga X_train y y_train guardados durante el entrenamiento.
    Son los datos con los que el modelo fue entrenado — la línea base.
    """
    import joblib
    X_train = joblib.load(f"{ARTIFACTS}/X_train.joblib")
    y_train = joblib.load(f"{ARTIFACTS}/y_train.joblib")

    df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    df["target"] = y_train.values
    return df

# ── Cargar datos de producción desde los logs ─────────────────────────────────
def load_current_data():
    """
    Lee las predicciones loggeadas por FastAPI.
    Estos son los datos reales que están llegando al modelo en producción.
    """
    if not os.path.isfile(LOGS):
        raise FileNotFoundError(
            f"No hay logs en {LOGS}. Haz al menos una predicción primero."
        )

    df = pd.read_csv(LOGS)

    # Renombrar 'prediction' a 'target' para que Evidently lo reconozca
    df = df.rename(columns={"prediction": "target"})

    # Codificar target igual que en entrenamiento (white=1, black=0)
    df["target"] = df["target"].map({"white": 1, "black": 0})

    # Quedarnos solo con las columnas relevantes
    cols = FEATURE_NAMES + ["target"]
    return df[[c for c in cols if c in df.columns]]

# ── Generar reporte ───────────────────────────────────────────────────────────
def generate_report():
    print("Cargando datos de referencia (entrenamiento)...")
    reference = load_reference_data()

    print("Cargando datos de producción (logs)...")
    current = load_current_data()

    print(f"  Referencia : {len(reference)} registros")
    print(f"  Producción : {len(current)} registros")

    if len(current) < 10:
        print("ADVERTENCIA: Menos de 10 predicciones en los logs.")
        print("El reporte puede no ser estadísticamente confiable.")

    # ── Reporte de drift ──────────────────────────────────────────────────────
    report = Report(metrics=[
        DataDriftPreset(),    # ¿Cambiaron las distribuciones de las features?
        DataQualityPreset(),  # ¿Hay valores nulos, outliers, o rangos inesperados?
    ])

    report.run(
        reference_data=reference,
        current_data=current
    )

    # ── Guardar reporte ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{REPORTS}/drift_report_{timestamp}.html"
    report.save_html(output_path)

    print(f"\nReporte guardado en: {output_path}")
    print("Ábrelo en tu navegador para ver el análisis completo.")
    return output_path

if __name__ == "__main__":
    generate_report()