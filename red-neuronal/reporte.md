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

**Precisión de la Red Neuronal (NN):** `64.88%`

### Red Neuronal vs Random Forest
- **Accuracy Random Forest:** `65.19%`
- **P-Value:** `0.88730`
- **Interpretación:** **No hay diferencia estadísticamente significativa**. La Red Neuronal logró un rendimiento estadísticamente equivalente al del Random Forest. Ambos modelos extrajeron la misma cantidad de información útil de estos datos.

### Red Neuronal vs Decision Tree
- **Accuracy Decision Tree:** `60.04%`
- **P-Value:** `0.00921`
- **Interpretación:** Existe una diferencia **estadísticamente significativa**. La Red Neuronal superó al Árbol de Decisión tradicional, logrando capturar patrones más complejos.

### Red Neuronal vs K-Nearest Neighbors
- **Accuracy K-NN:** `60.97%`
- **P-Value:** `0.01409`
- **Interpretación:** Existe una diferencia **estadísticamente significativa**. La Red Neuronal fue consistentemente mejor que el modelo K-NN para predecir el resultado de las partidas.

---

## 4. Conclusión Final

**¿La red neuronal realmente mejoró algo o los modelos clásicos eran suficientes para este problema?**

La conclusión tras reentrenar y evaluar rigurosamente es muy reveladora: **La Red Neuronal superó estadísticamente a modelos más simples (Decision Tree y K-NN), y logró empatar (no hay diferencia significativa) con el ensamble clásico más robusto (Random Forest).**

**¿Qué significa esto para el proyecto?**
1. **El límite de los datos:** Tanto la Red Neuronal (~64.9%) como el Random Forest (~65.2%) chocaron contra una pared en la precisión. Esto nos indica que las características actuales del dataset no contienen suficiente "señal" predictiva para adivinar al ganador de una partida de ajedrez con mayor exactitud (el ajedrez es altamente complejo y factores como el ranking o la apertura, por sí solos, no son determinantes absolutos).
2. **Costo-Beneficio:** Aunque la Red Neuronal logró igualar al mejor modelo clásico, el esfuerzo computacional y de diseño (diagnosticar overfitting, aplicar Dropout, calibrar capas) fue considerable. Para datos tabulares de este tipo, el **Random Forest** suele ser la opción más pragmática al ofrecer el máximo rendimiento "fuera de la caja".

El ejercicio cumplió excelentemente su objetivo analítico: demostró empíricamente (vía Test de McNemar y TensorBoard) cómo comparar formalmente un modelo de *Deep Learning* contra *Machine Learning* clásico, probando que modelos totalmente distintos pueden converger exactamente en el mismo límite de aprendizaje dictado por la calidad de los datos.
