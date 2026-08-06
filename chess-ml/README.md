# Chess-ML API 🚀
### Fase 3 — Despliegue y Monitoreo en Producción

Servicio de predicción del ganador de partidas de ajedrez mediante una red neuronal entrenada sobre datos de [Lichess](https://lichess.org/). Esta fase toma el mejor modelo encontrado en la Fase 2 (Grid Search con MLflow) y lo convierte en un microservicio consumible, contenerizado con Docker y monitoreado con Evidently.

---

## Contexto del proyecto

| Fase | Descripción | Ubicación |
|------|-------------|-----------|
| **Fase 1** | Entrenamiento de red neuronal, diagnóstico de overfitting y evaluación estadística (McNemar) | `Red-Neuronal/` |
| **Fase 2** | Grid Search con MLflow, análisis factorial 2⁴ y pipeline reproducible con DVC | `Red-Neuronal/` |
| **Fase 3** | Despliegue con FastAPI + Docker y monitoreo de drift con Evidently | `chess-ml/` ← estás aquí |

---

## Arquitectura

```
Usuario / App
     │
     ▼  POST /predict
┌─────────────────┐
│    FastAPI      │  ← valida entrada, predice, loggea
│   (Docker)      │  ← contenedor aislado y reproducible
└────────┬────────┘
         │ loggea cada request
         ▼
┌─────────────────┐
│  logs/          │
│  predictions    │  ← CSV con inputs y predicciones
│  .csv           │
└────────┬────────┘
         │ Evidently compara vs datos de entrenamiento
         ▼
┌─────────────────┐
│  reports/       │
│  drift_report   │  ← HTML con análisis de drift por feature
│  _*.html        │
└─────────────────┘
```

---

## Modelo desplegado

El modelo es la mejor configuración encontrada en el Grid Search de la Fase 2, validada mediante análisis factorial 2⁴ con repetición de semillas:

| Hiperparámetro | Valor | Importancia |
|----------------|-------|-------------|
| `hidden_size` | 128 | ✅ Efecto real |
| `dropout` | 0.3 | ✅ Efecto real |
| `lr` | 0.0005 | — Ruido en el rango probado |
| `weight_decay` | 0.0001 | — Ruido en el rango probado |

**Accuracy en test:** 66.22% · **Baseline (Random Forest):** 65.19% (diferencia no significativa, p=0.887)

**Features utilizadas (seleccionadas por RFE):**

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `white_rating` | Continua (norm.) | Rating ELO del jugador de blancas |
| `black_rating` | Continua (norm.) | Rating ELO del jugador de negras |
| `opening_ply` | Continua (norm.) | Número de jugadas en la apertura |
| `victory_status_mate` | One-Hot | La partida terminó en jaque mate |
| `victory_status_resign` | One-Hot | La partida terminó en rendición |
| `opening_eco_B00` | One-Hot | Apertura: Defensa Pirc/Owen |
| `opening_eco_B01` | One-Hot | Apertura: Defensa Escandinava |
| `opening_eco_C00` | One-Hot | Apertura: Apertura Francesa |
| `opening_eco_C41` | One-Hot | Apertura: Defensa Philidor |
| `opening_eco_D00` | One-Hot | Apertura: Apertura de Peón de Dama |
| `increment_code_10+0` | One-Hot | Control de tiempo: 10 min sin incremento |
| `increment_code_15+0` | One-Hot | Control de tiempo: 15 min sin incremento |

> **Nota sobre valores One-Hot:** solo una variable del mismo grupo puede ser 1 a la vez. Si todas son 0, se asume la categoría menos frecuente (filtrada en preprocesamiento).

---

## Estructura del proyecto

```
chess-ml/
└── api/
    ├── artifacts/              # Artefactos del modelo (gestionados por DVC)
    │   ├── chess_nn_best.pt    # Pesos de la red neuronal (.gitignore)
    │   ├── scaler.joblib       # StandardScaler ajustado sobre X_train (.gitignore)
    │   ├── feature_names.json  # Lista ordenada de features seleccionadas por RFE
    │   ├── X_train.joblib      # Datos de referencia para Evidently (.gitignore)
    │   └── y_train.joblib      # Etiquetas de referencia para Evidently (.gitignore)
    ├── logs/                   # Predicciones loggeadas en producción (.gitignore)
    │   └── predictions.csv
    ├── reports/                # Reportes de drift generados por Evidently (.gitignore)
    │   └── drift_report_*.html
    ├── Dockerfile              # Definición del contenedor
    ├── main.py                 # FastAPI — endpoints de predicción y monitoreo
    ├── monitoring.py           # Generación de reportes de drift con Evidently
    └── requirements.txt        # Dependencias del servicio
```

---

## Instalación y uso

### Opción A — Con Docker (recomendado)

**Requisitos:** Docker instalado y corriendo.

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd chess-ml/api

# 2. Construir la imagen
docker build -t chess-api .

# 3. Correr el contenedor
docker run -p 8000:8000 chess-api
```

La API queda disponible en `http://localhost:8000`.

### Opción B — Local (sin Docker)

```bash
cd chess-ml/api
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Información del servicio y lista de features esperadas |
| `GET` | `/health` | Estado del servicio y confirmación de modelo cargado |
| `GET` | `/example` | Ejemplo de request válido con valores realistas |
| `POST` | `/predict` | Predicción del ganador dado un conjunto de features |
| `POST` | `/monitoring/report` | Genera reporte de drift comparando logs vs entrenamiento |

### Documentación interactiva

FastAPI genera automáticamente una interfaz visual en:

```
http://localhost:8000/docs
```

Desde ahí puedes probar todos los endpoints directamente desde el navegador sin necesidad de herramientas externas.

---

## Ejemplo de uso

### Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "white_rating": 0.65,
         "black_rating": 0.40,
         "opening_ply": 0.30,
         "victory_status_mate": 0.0,
         "victory_status_resign": 1.0,
         "opening_eco_B00": 0.0,
         "opening_eco_B01": 0.0,
         "opening_eco_C00": 1.0,
         "opening_eco_C41": 0.0,
         "opening_eco_D00": 0.0,
         "increment_code_10+0": 1.0,
         "increment_code_15+0": 0.0
       }
     }'
```

> El ejemplo representa una partida donde las blancas tienen rating significativamente mayor (0.65 vs 0.40), la partida terminó en rendición, con apertura francesa y control de tiempo de 10 minutos.

### Response

```json
{
  "winner": "white",
  "confidence": 0.7143,
  "probabilities": {
    "black": 0.2857,
    "white": 0.7143
  }
}
```

---

## Monitoreo de drift

Cada predicción queda registrada automáticamente en `logs/predictions.csv`. Para generar un reporte de salud del modelo:

### Desde la API

```bash
curl -X POST "http://localhost:8000/monitoring/report"
```

### Extraer el reporte del contenedor

```bash
# Obtener el ID del contenedor
docker ps

# Copiar los reportes a tu máquina local
docker cp <container_id>:app/reports/ ./reports/
```

Abre el HTML generado en tu navegador — Evidently muestra:

- **Data Drift:** distribución de cada feature en producción vs entrenamiento
- **Data Quality:** valores fuera de rango, nulos o con varianza cero
- **Resumen:** porcentaje de columnas con drift detectado

### Interpretación del umbral

| % columnas con drift | Interpretación |
|----------------------|----------------|
| < 30% | ✅ Modelo estable |
| 30% – 60% | ⚠️ Revisar features con mayor drift score |
| > 60% | 🔴 Considerar reentrenamiento |

> **Nota:** un reporte con pocas predicciones de prueba puede mostrar drift artificial. El monitoreo es confiable con un mínimo de 100 registros en producción.

---

## Reentrenamiento

Si Evidently detecta drift significativo o el accuracy en producción cae, el flujo de reentrenamiento es:

```bash
# 1. Actualizar el dataset en Red-Neuronal/
# 2. Reproducir el pipeline de Fase 2
cd Red-Neuronal/
dvc repro

# 3. Exportar los nuevos artefactos a chess-ml/api/artifacts/
# 4. Reconstruir la imagen Docker
cd ../chess-ml/api/
docker build -t chess-api .
```

---

## Tecnologías

| Herramienta | Uso |
|-------------|-----|
| **FastAPI** | Framework para el servicio REST |
| **Uvicorn** | Servidor ASGI de producción |
| **PyTorch** | Carga e inferencia del modelo |
| **scikit-learn** | Scaler para preprocesamiento de inputs |
| **Docker** | Contenerización del servicio |
| **Evidently** | Monitoreo de drift en producción |
| **Pydantic** | Validación automática de datos de entrada |

---

## Autor

**Álvaro Alexander Velázquez Matus**
Estudiante de Ingeniería en Inteligencia Artificial — ESCOM, IPN
[github.com/AlvaroMkoko](https://github.com/AlvaroMkoko)