# Predicción de resultado de partidas de ajedrez — Red Neuronal vs Modelos Clásicos

Este proyecto entrena una red neuronal en PyTorch para predecir el ganador de una partida de ajedrez (blancas o negras) a partir de metadatos de la partida, y la compara estadísticamente contra tres modelos clásicos de Machine Learning (Random Forest, Decision Tree, K-Nearest Neighbors). Incluye un grid search con análisis estadístico de hiperparámetros y un pipeline reproducible con DVC.

## Estructura del proyecto

```
.
├── games.csv                          # Dataset original de partidas
├── Funciones.py                       # Preprocesamiento compartido (filtrado, codificación, balanceo)
├── chess_random_forest_model.joblib   # Modelo RF original, usado para recuperar features vía RFE
└── red-neuronal/
    ├── entrenamiento.ipynb            # Notebook principal: EDA, overfitting, grid search, McNemar, DVC
    ├── reporte.md                     # Reporte de resultados y comparación estadística final
    ├── params.yaml                    # Hiperparámetros del modelo final (entrada del pipeline DVC)
    ├── data/
    │   └── processed.pkl              # Datos preprocesados (generado por preprocess.py)
    ├── models/
    │   └── best_model.pth             # Pesos del modelo final (generado por train_best.py)
    ├── metrics/
    │   └── scores.json                # Métricas finales (generado por evaluate.py)
    └── scripts/
        ├── preprocess.py              # Etapa 1 del pipeline DVC
        ├── train_best.py              # Etapa 2 del pipeline DVC
        └── evaluate.py                # Etapa 3 del pipeline DVC
```

## Qué se hizo

1. **Preprocesamiento**: carga de `games.csv`, filtrado de categorías poco frecuentes (`increment_code`, `opening_eco`), codificación de variables categóricas, balanceo de clases y selección de features mediante RFE (heredada del Random Forest original). Split train/test hecho **antes** de ajustar el `StandardScaler`, para evitar data leakage.

2. **Modelos clásicos (baseline)**: Random Forest, Decision Tree y K-Nearest Neighbors reentrenados desde cero solo sobre el train set, para tener una comparación justa contra la red neuronal.

3. **Red neuronal con overfitting diagnosticado**: una primera arquitectura sin regularización mostró sobreajuste claro en TensorBoard (loss de validación subiendo mientras la de entrenamiento seguía bajando).

4. **Mitigación del overfitting**: se agregaron Dropout, Weight Decay (L2) y Batch Normalization. La brecha train/validación se redujo notablemente.

5. **Grid search con análisis estadístico riguroso** (ver `entrenamiento.ipynb`, sección de MLflow): se probó un diseño factorial 2⁴ sobre `lr`, `hidden_size`, `dropout` y `weight_decay` (16 combinaciones), registrado en MLflow. Para separar señal de ruido de entrenamiento se corrieron **5 semillas por combinación** (80 corridas totales) y se calcularon efectos principales con su error estándar. Conclusión: `dropout` y `hidden_size` tienen efecto real y estadísticamente distinguible del ruido; `lr` y `weight_decay`, en el rango probado, no.

6. **Comparación estadística final (Test de McNemar)**: se compararon las predicciones de la red neuronal contra cada modelo clásico sobre el mismo test set.

7. **Pipeline reproducible con DVC**: las 3 etapas (`preprocess` → `train_best` → `evaluate`) están encadenadas vía `dvc.yaml`, de forma que cambiar un valor en `params.yaml` y correr `dvc repro` solo re-ejecuta las etapas afectadas.

## Configuración final del modelo

Según `params.yaml`:

| Hiperparámetro | Valor |
|---|---|
| `lr` | 0.0005 |
| `dropout` | 0.3 |
| `hidden_size` | 128 |
| `weight_decay` | 0.001 |
| `epochs` | 120 |
| `batch_size` | 64 |

## Resultados

| Modelo | Accuracy | vs. Red Neuronal (p-value McNemar) | ¿Diferencia significativa? |
|---|---|---|---|
| Red Neuronal | 64.88% | — | — |
| Random Forest | 65.19% | 0.887 | No |
| Decision Tree | 60.04% | 0.009 | Sí |
| K-Nearest Neighbors | 60.97% | 0.014 | Sí |

**Conclusión principal:** la red neuronal superó a los modelos simples (Decision Tree, K-NN) pero empató estadísticamente con el Random Forest. Esto sugiere que el techo de accuracy (~65%) está determinado por la señal disponible en los datos, no por la elección de modelo — el ajedrez tiene demasiados factores no capturados en estos features como para predecir el ganador con mucha más precisión.

## Cómo reproducir

```bash
cd red-neuronal

# Ejecutar el pipeline completo (preprocess -> train -> evaluate)
dvc repro

# Ver métricas finales
dvc metrics show
```

Para probar otra configuración de hiperparámetros, edita `params.yaml` (por ejemplo `lr` o `hidden_size`) y vuelve a correr `dvc repro`: DVC detecta el cambio y solo re-ejecuta las etapas `train` y `evaluate`, dejando `preprocess` cacheado.

## Notas de diseño

- El scaler (`StandardScaler`) se ajusta únicamente sobre el train set y se aplica (sin reajustar) al test set, evitando fuga de información.
- Los modelos clásicos se reentrenaron desde cero en este pipeline en lugar de cargar los artefactos serializados del proyecto original, ya que esos habían sido entrenados sobre el dataset completo (incluyendo lo que aquí es test set), lo que inflaba artificialmente su accuracy.
- El análisis de hiperparámetros usó repeticiones con semilla controlada específicamente para poder distinguir efectos reales de la variabilidad inherente al entrenamiento (inicialización de pesos, orden de batches, dropout estocástico).