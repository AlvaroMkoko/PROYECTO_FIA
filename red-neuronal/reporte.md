# Reporte de Entrenamiento: Red Neuronal vs Modelos Clásicos

Este documento resume los resultados del entrenamiento de una red neuronal en PyTorch para predecir el resultado de partidas de ajedrez, diagnosticando y mitigando el sobreajuste (overfitting), y comparando estadísticamente el resultado final con tres modelos clásicos previamente entrenados: Random Forest, K-Nearest Neighbors y Decision Tree.

## 1. Diagnóstico de Overfitting con TensorBoard

![Pérdida en Entrenamiento y Validación (Overfitting)](resultados_experimentos/loss_overfit.png)
![Precisión en Entrenamiento y Validación (Overfitting)](resultados_experimentos/accuracy_overfit.png)

**Análisis:**
- **¿Hubo overfitting inicial?:** Sí. 
- **¿Cómo se detectó?:** Al observar las curvas en TensorBoard, se puede apreciar que la pérdida de entrenamiento (`Train Loss`) continúa descendiendo hacia cero a lo largo de las épocas, mientras que la pérdida de validación (`Validation Loss`) llega a un punto mínimo y luego comienza a subir. Esto indica que la red estaba memorizando los datos de entrenamiento y perdiendo capacidad de generalización frente a datos nuevos.

## 2. Mitigación de Overfitting

![Pérdida en Entrenamiento y Validación (Regularizado)](resultados_experimentos/loss_reg.png)
![Precisión en Entrenamiento y Validación (Regularizado)](resultados_experimentos/accuracy_reg.png)

**Técnicas aplicadas:**
- **Dropout (0.5 y 0.3):** Se añadieron capas de Dropout para "apagar" un porcentaje de neuronas aleatoriamente en cada iteración, forzando a la red a no depender de características específicas y aprender patrones más distribuidos.
- **Weight Decay (Regularización L2):** Se añadió al optimizador Adam (`weight_decay=1e-4`) para penalizar pesos muy grandes, evitando que la red se ajuste a ruidos en los datos.
- **Batch Normalization:** Para estabilizar el aprendizaje y mejorar la convergencia.

**Resultado:** 
Al observar la nueva curva en TensorBoard, la divergencia entre la pérdida de entrenamiento y validación se redujo significativamente. La pérdida de validación dejó de dispararse hacia arriba, logrando una mejor generalización, aunque revelando el límite de aprendizaje de nuestra arquitectura sobre estos datos tabulares.

---

## 3. Comparación Estadística (Test de McNemar)

Comparamos las predicciones en el conjunto de prueba (Test Set) para verificar si las diferencias en precisión (*Accuracy*) entre la Red Neuronal (NN) y los modelos clásicos son estadísticamente significativas (p-value < 0.05).

**Precisión de la Red Neuronal (NN):** `65.50%`

### Red Neuronal vs Random Forest
- **Accuracy Random Forest:** `93.51%`
- **P-Value:** `0.00000`
- **Interpretación:** Existe una diferencia **estadísticamente significativa**. El modelo de Random Forest es abrumadoramente superior a la Red Neuronal para este problema.

### Red Neuronal vs Decision Tree
- **Accuracy Decision Tree:** `91.86%`
- **P-Value:** `0.00000`
- **Interpretación:** Existe una diferencia **estadísticamente significativa**. Al igual que el Random Forest, el Árbol de Decisión tradicional superó ampliamente a la Red Neuronal.

### Red Neuronal vs K-Nearest Neighbors
- **Accuracy K-NN:** `52.42%`
- **P-Value:** `0.00000`
- **Interpretación:** Existe una diferencia **estadísticamente significativa**. En este caso, la Red Neuronal superó al K-NN, el cual obtuvo un rendimiento cercano al azar.

---

## 4. Conclusión Final

**¿La red neuronal realmente mejoró algo o los modelos clásicos eran suficientes para este problema?**

La conclusión es contundente: **Los modelos clásicos basados en árboles (Random Forest y Decision Tree) fueron muy superiores y más que suficientes para este problema.**

**¿Por qué ocurrió esto?**
1. **Naturaleza de los Datos:** Nuestro conjunto de datos consta de variables puramente tabulares (rankings, aperturas codificadas, tiempos). Las redes neuronales multicapa (MLP) estándar suelen tener dificultades para superar a los ensambles de árboles (como Random Forest o XGBoost) en datos tabulares sin arquitecturas hiper-especializadas (como embeddings categóricos).
2. **Eficiencia:** El Random Forest logró casi un 94% de precisión con menor esfuerzo de calibración (tuning), mientras que la red neuronal requirió lidiar con la regularización, normalización de datos y ajuste de hiperparámetros solo para alcanzar un modesto 65.5%. 

El ejercicio cumplió su objetivo analítico: demostró empíricamente (vía Test de McNemar y TensorBoard) que aplicar *Deep Learning* no siempre es la solución óptima, y que un entendimiento sólido de los algoritmos clásicos sigue siendo invaluable.
