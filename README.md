# Proyecto de Procesamiento, Selección de Características y Clasificación en Partidas de Ajedrez

Este repositorio contiene un sistema completo de procesamiento de datos, selección de características y clasificación predictiva de resultados en partidas de ajedrez (Lichess) mediante algoritmos de Machine Learning en Python, complementado con una interfaz gráfica interactiva en Flet.

El proyecto ha sido desarrollado como el **Proyecto Final** para la asignatura de **Fundamentos de Inteligencia Artificial**.

---

## Participantes y Docente

* **Integrantes del Equipo**:
  * **Bustillos Cruz Jonatan**
  * **Martínes Contreras Leonardo**
  * **Salazar Bravo Alejandro Román**
  * **Velazquez Matus Álvaro Alexander**
* **Profesor**:
  * **Hernández Cruz Macario**

---

## Introducción

El sistema procesa registros de partidas de ajedrez provenientes del archivo [games.csv](games.csv), realiza tareas de limpieza, balanceo y codificación de variables, y aplica selección recursiva de características (RFE) mediante Bosques Aleatorios. Con este subconjunto optimizado de características, el proyecto entrena y evalúa clasificadores para predecir el resultado de la partida:

1. **Ganador del encuentro** (Blancas `1` / Negras `0`).

El sistema también proporciona una interfaz visual e interactiva que permite a los usuarios modificar las características de una partida y visualizar la predicción en tiempo real.

---

## Características principales

* **Limpieza y preprocesamiento de datos**: Filtrado automático de partidas que terminaron en empate (`draw`) para centrar el modelo en la clasificación binaria (victoria de blancas o negras).
* **Filtrado de valores infrecuentes**: Limpieza de categorías con presencia menor al 2% o menos de 100 registros en variables nominales como códigos de incremento (`increment_code`) y aperturas (`opening_eco`).
* **Codificación y Escalado de Variables**:
  * **Datos Discretos**: Normalización de ratings de jugadores y jugadas de apertura (`white_rating`, `black_rating`, `opening_ply`) con `MinMaxScaler`.
  * **Datos Nominales**: Aplicación de One-Hot Encoding en el estado de victoria (`victory_status`), código de apertura (`opening_eco`) e incremento (`increment_code`), y `LabelEncoder` para la variable objetivo `winner`.
* **Balanceo del Dataset**: Submuestreo de la clase mayoritaria en la variable objetivo (`winner`) para lograr una proporción de 50/50 y evitar sesgos en el entrenamiento.
* **Selección de características mediante RFE**: Implementación de Eliminación Recursiva de Características (RFE) combinada con un *Random Forest Classifier*, optimizando el área bajo la curva ROC (ROC AUC).
* **Clasificación y entrenamiento automatizado**:
  * *Random Forest Classifier*: Utilizado en el proceso RFE y guardado del modelo base.
  * *K-Nearest Neighbors (K-NN)*: Clasificación no paramétrica basada en vecinos cercanos, estimando automáticamente el número óptimo $k \approx \sqrt{N}$ (ajustado a impar) y reduciendo la dimensionalidad a 2D mediante PCA para graficar la dispersión.
  * *Decision Tree Classifier*: Clasificador basado en árboles de decisión utilizando el criterio de impureza Gini y exportación del gráfico del árbol jerárquico.
* **Interfaz de Usuario Interactiva (Flet)**: Una aplicación moderna y de alto rendimiento que incluye:
  * Pestañas independientes para comparar el rendimiento de los modelos K-NN y Árbol de Decisión.
  * Visualización interactiva de métricas de rendimiento y matrices de confusión/gráficas de dispersión.
  * Formulario con deslizadores (sliders) y selectores interactivos para realizar predicciones dinámicas en tiempo real.

---

## Requisitos

Para ejecutar correctamente el proyecto, se recomienda utilizar **Python 3.9 o superior**.

### Instalación de dependencias

Instala los requerimientos recomendados ejecutando el siguiente comando en la consola:

```bash
pip install numpy pandas scikit-learn joblib flet matplotlib torch tensorboard statsmodels jupyter
```

---

## Dependencias utilizadas

| Librería | Uso principal |
| :--- | :--- |
| `numpy` | Operaciones matemáticas rápidas y manipulación de matrices numéricas. |
| `pandas` | Carga de datos de partidas ([games.csv](games.csv)), estructuración en DataFrames y codificación (dummies). |
| `scikit-learn` | Modelos de Machine Learning (Random Forest, K-NN, Decision Tree), RFE para selección, MinMaxScaler, StandardScaler, PCA y métricas de evaluación. |
| `joblib` | Serialización y carga rápida de modelos y nombres de características entrenados (`.joblib`). |
| `flet` | Construcción de la interfaz gráfica interactiva del usuario (GUI) con pestañas y controles interactivos. |
| `matplotlib` | Graficación de curvas de RFE, matrices de confusión, diagramas de dispersión 2D y el árbol de decisión. |
| `torch` | Entrenamiento de la Red Neuronal (MLP) en PyTorch, incluyendo los bucles de entrenamiento y validación. |
| `tensorboard` | Visualización de las curvas de pérdida y precisión para el diagnóstico de overfitting. |
| `statsmodels` | Implementación del Test de McNemar para la comparación estadística entre modelos. |
| `jupyter` | Ejecución del notebook interactivo `red-neuronal/entrenamiento.ipynb`. |

---

## Estructura del repositorio

```text
├── img/                     # Gráficas generales del sistema
│   ├── Figure_1.png
│   └── Figure_matriz.png
├── img2/                    # Matrices de confusión y gráficas de dispersión para la interfaz
│   ├── Grafica_dispersion_knn.png
│   ├── matriz_confusion_knn.png
│   └── matriz_confusion_tree.png
├── red-neuronal/            # Experimento de Deep Learning con PyTorch
│   ├── entrenamiento.ipynb  # Notebook con todo el proceso de entrenamiento y evaluación
│   ├── reporte.md           # Reporte detallado: overfitting, regularización y McNemar
│   ├── resultados_experimentos/  # Capturas de TensorBoard (curvas de loss y accuracy)
│   └── runs/                # Logs de TensorBoard generados durante el entrenamiento
├── games.csv                # Dataset original de partidas de ajedrez en formato CSV
├── Funciones.py             # Módulo con funciones de preprocesamiento, selección y modelado
├── main.py                  # Script principal para limpieza de datos, selección de características y entrenamiento
├── visual.py                # Aplicación GUI interactiva desarrollada con Flet
├── metrics_report_knn.txt   # Reporte con métricas detalladas del modelo K-NN
├── tree_metrics_report.txt  # Reporte con métricas detalladas del Árbol de Decisión
├── arbol_de_decision_diabetes.png # Gráfico del árbol de decisión jerárquico exportado
├── chess_random_forest_model.joblib # Serialización de Random Forest y nombres de mejores características
├── k_nn_model.joblib        # Serialización del modelo K-NN optimizado
├── decision_tree_model.joblib # Serialización del modelo Árbol de Decisión
└── README.md                # Documentación del proyecto (este archivo)
```

---

## Guía de uso

### 1. Preparación de datos y entrenamiento de modelos

Para procesar el conjunto de datos de partidas [games.csv](games.csv), codificar las variables, seleccionar las mejores características con RFE e inicializar el entrenamiento de los modelos, ejecuta:

```bash
python main.py
```

> 💡 **Nota de entrenamiento**: Si deseas volver a entrenar y evaluar los modelos K-NN y de Árbol de Decisión para regenerar los reportes de métricas y guardarlos como archivos `.joblib`, puedes abrir el archivo [main.py](main.py) y descomentar las líneas 64 y 65:
> ```python
> f.K_NN(data)
> f.Tree(data)
> ```

Al ejecutarse, el sistema:
1. Filtrará empates y limpiará valores atípicos.
2. Codificará variables mediante `MinMaxScaler` y One-Hot Encoding.
3. Balanceará el número de victorias entre Blancas y Negras.
4. Ejecutará RFE para seleccionar el subconjunto óptimo de características.
5. Guardará el modelo base y los nombres de las características seleccionadas en [chess_random_forest_model.joblib](chess_random_forest_model.joblib).
6. Entrenará y guardará los modelos [k_nn_model.joblib](k_nn_model.joblib) y [decision_tree_model.joblib](decision_tree_model.joblib), generando simultáneamente los archivos de reporte de métricas y gráficos en `img2/`.

### 2. Ejecución de la Interfaz Gráfica (Flet)

Una vez que los modelos han sido entrenados y sus archivos `.joblib` se encuentran en el directorio raíz, puedes lanzar la interfaz interactiva:

```bash
python visual.py
```

Desde esta GUI, podrás:
* **Explorar el rendimiento**: Consultar la matriz de confusión y la gráfica de dispersión 2D para el modelo K-NN, y la matriz de confusión del Árbol de Decisión, así como sus reportes de exactitud, precisión, sensibilidad y F1-score correspondientes en [metrics_report_knn.txt](metrics_report_knn.txt) y [tree_metrics_report.txt](tree_metrics_report.txt).
* **Realizar predicciones**: Configurar en tiempo real el rating estimado del jugador de piezas blancas y negras, el número de jugadas en la fase de apertura, el método de conclusión de la partida, y el código de apertura. Al presionar **Predicción**, el sistema cargará los datos, los transformará al espacio de características seleccionado e indicará si la victoria se predice para las **Blancas** o las **Negras**.

---

## Resultados y Métricas del Modelo

Durante la evaluación de los modelos clasificados utilizando un split de prueba del 20% sobre el dataset balanceado de partidas, se obtuvieron las siguientes métricas de rendimiento:

### 1. Modelo K-NN ($k$-vecinos más cercanos)
* **Exactitud (Accuracy)**: `62.41%`
* **Precisión (Precision)**: `60.43%`
* **Sensibilidad (Recall)**: `60.82%`
* **Puntuación F1 (F1 Score)**: `60.63%`
* **Área bajo la curva (ROC AUC)**: `62.34%`
* **Detalle de la Matriz de Confusión**:
  * Verdaderos Negativos (TN): 325 (Derrotas de blancas/victorias de negras clasificadas correctamente)
  * Verdaderos Positivos (TP): 281 (Victorias de blancas clasificadas correctamente)
  * Falsos Positivos (FP): 184
  * Falsos Negativos (FN): 181

### 2. Modelo Árbol de Decisión
* **Exactitud (Accuracy)**: `61.48%`
* **Precisión (Precision)**: `58.98%`
* **Sensibilidad (Recall)**: `62.55%`
* **Puntuación F1 (F1 Score)**: `60.71%`
* **Área bajo la curva (ROC AUC)**: `61.53%`
* **Detalle de la Matriz de Confusión**:
  * Verdaderos Negativos (TN): 308
  * Verdaderos Positivos (TP): 289
  * Falsos Positivos (FP): 201
  * Falsos Negativos (FN): 173

### 3. Modelo Red Neuronal (PyTorch MLP)
* **Exactitud (Accuracy)**: `64.88%`

> Ver el reporte completo en [`red-neuronal/reporte.md`](red-neuronal/reporte.md).

---

## Red Neuronal con PyTorch: Diagnóstico y Comparación

Como extensión analítica del proyecto, se entrenó una Red Neuronal Multicapa (MLP) en PyTorch sobre el mismo conjunto de datos y con el mismo preprocesamiento, con el objetivo de:
1. Aprender a diagnosticar **overfitting** mediante curvas de pérdida en TensorBoard.
2. Aplicar técnicas de **regularización** para mitigarlo.
3. Evaluar de forma **estadísticamente rigurosa** si la red neuronal supera a los modelos clásicos.

### Diagnóstico de Overfitting (TensorBoard)

Se entrenó primero una red intencionalmente grande y sin regularización (capas de 512-512-256 neuronas). Al visualizar las curvas en TensorBoard se observó el overfitting clásico: la pérdida de entrenamiento (`Train Loss`) descendió hasta casi cero mientras que la pérdida de validación (`Validation Loss`) llegó a un mínimo y luego comenzó a subir, indicando memorización y pérdida de generalización.

### Mitigación: Regularización

Se rediseñó la arquitectura aplicando tres técnicas:

| Técnica | Función |
| :--- | :--- |
| **Dropout** (0.5 / 0.3) | Desactiva neuronas al azar en cada iteración, evitando codependencia entre ellas. |
| **Weight Decay** (`1e-4`) | Penalización L2 en el optimizador Adam para reducir el sobreajuste a ruido. |
| **Batch Normalization** | Estabiliza el aprendizaje y acelera la convergencia entre capas. |

Tras la regularización, las curvas de validación dejaron de divergir, logrando una mejor generalización.

### Comparación Estadística (Test de McNemar)

Se compararon las predicciones individuales de la Red Neuronal vs. cada modelo clásico sobre el mismo conjunto de prueba usando el **Test de McNemar** (p-value < 0.05 = diferencia significativa).

| Comparación | Accuracy NN | Accuracy Modelo Clásico | P-Value | Conclusión |
| :--- | :---: | :---: | :---: | :--- |
| NN vs. Random Forest | `64.88%` | `65.19%` | `0.887` | ✅ Sin diferencia significativa — rendimiento equivalente |
| NN vs. K-NN | `64.88%` | `60.97%` | `0.014` | ⚠️ NN significativamente superior |
| NN vs. Decision Tree | `64.88%` | `60.04%` | `0.009` | ⚠️ NN significativamente superior |

### Conclusión

La Red Neuronal **superó estadísticamente** a los modelos más simples (K-NN y Árbol de Decisión) y **empató en rendimiento** con el Random Forest (sin diferencia estadística). Todos los modelos basados en las características disponibles convergen alrededor del mismo límite de ~65%, lo que sugiere que la información contenida en el dataset (ratings, apertura, tiempo) no es suficiente para predecir el ganador con mayor exactitud — el resultado de una partida de ajedrez depende de factores intangibles no capturados en estas columnas.

Desde una perspectiva de **costo-beneficio**, el Random Forest es la opción más pragmática: logra el mismo nivel de precisión con mucho menor esfuerzo de diseño y calibración que una red neuronal.

> 📄 Reporte completo con gráficas de TensorBoard incluido en [`red-neuronal/reporte.md`](red-neuronal/reporte.md).

---

## Notas importantes

* **Archivo de imagen del Árbol de Decisión**: Por motivos heredados de la plantilla de visualización, la imagen jerárquica del árbol de decisión entrenado se guarda bajo el nombre de [arbol_de_decision_diabetes.png](arbol_de_decision_diabetes.png).
* **Sincronización de características**: La selección de características a través del archivo [chess_random_forest_model.joblib](chess_random_forest_model.joblib) define de forma dinámica qué columnas del DataFrame preprocesado se usarán para entrenar a K-NN y al Árbol de Decisión. Asegúrate de entrenar primero el selector si alteras el preprocesamiento de datos en [Funciones.py](Funciones.py).
* **Exclusión de Tablas**: El proyecto está configurado para la predicción binaria del ganador (Blancas vs. Negras), por lo que excluye todas las partidas que concluyeron en tablas (`draw`) de forma automática en la función `get_data` de [Funciones.py](Funciones.py).

---

## Posibles aplicaciones

* **Sistemas de recomendación y predicción**: Herramienta interactiva para que jugadores amateurs estimen sus probabilidades de ganar según su rating y el tipo de apertura elegido.
* **Análisis demográfico de partidas**: Visualización estadística en 2D (a través de PCA) de cómo se distribuyen los ganadores según las variables clave de la partida.
* **Propósito académico**: Demostración de técnicas clave de Inteligencia Artificial (procesamiento, codificación nominal/discreta, balanceo de datos, selección RFE, reducción de dimensionalidad PCA, evaluación clásica de modelos, desarrollo de interfaces modernas y análisis empírico de Deep Learning con PyTorch incluyendo diagnóstico de overfitting y comparación estadística rigurosa).