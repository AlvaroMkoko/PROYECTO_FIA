import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,  LabelEncoder
import Funciones as f


import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import sys
import random

import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


def get_data(name):
    data  = pd.read_csv(name)
    # print(data.head())
    data = data.loc[:,['victory_status','winner','increment_code','white_rating','black_rating','opening_eco','opening_ply']]
    #  Eliminar los datos de la columna victory_status 'draw'
    data = data[data['victory_status'] != 'draw']
    data = data[data['winner'] != 'draw']
    # data = data[data['victory_status'] != 'outoftime']
    # print(len(data))

    return data

def balance(data):
    """
    Ahora hay que hacer que las clases objetivo tengan el mismo numero de datos
    """
    # n_black = len(data[data['winner'] == 'black'])
    # n_white = len(data[data['winner'] == 'white'])

    # Contar cuántos elementos hay por clase
    conteo_clase = data['winner'].value_counts()
    # Encontrar el tamaño de la clase minoritaria
    minimo_clase = conteo_clase.min()

    # Submuestreo de cada clase
    data = data.groupby('winner', group_keys=False).apply(lambda x: x.sample(minimo_clase)).reset_index(drop=True)
    print("Negras: " + str(len(data[data['winner'] == 0])))
    print("Blancas: " + str(len(data[data['winner'] == 1])))
    return data

def code(data):
    """
        Tenemos principalmnte datos cuantitatios discretos y cualitativas nominales.
        TIPOS
        Discretos: turns, rating, white_rating, black_rating, opening_ply
        Nominales: victory_status, winner, opening_eco, increment_code
    """
    # Para datos discretos
    discrete_columns = ['white_rating', 'black_rating', 'opening_ply']
    scaler = MinMaxScaler()
    data[discrete_columns] = scaler.fit_transform(data[discrete_columns])

    # para datos nominales
    nominal_columns = ['victory_status', 'opening_eco', 'increment_code']


    # Aplicar One-Hot Encoding
    data = pd.get_dummies(data, columns=nominal_columns)
    label = LabelEncoder()
    data['winner'] = label.fit_transform(data['winner'])
    for original, encoded in zip(label.classes_, range(len(label.classes_))):
        print(f"-----------------{original} -> {encoded}------------------------")
    # Los discretos se noralizan a [0,1]
    # Los nominales se normalizan con One-Hot Encoding
    column_to_move = 'winner'
    data[column_to_move] = data.pop(column_to_move)
    print(data)
    # print(data)
    X = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return data

def filter(df, column, min_proportion=0.01, min_games=100):
    # Contar frecuencias
    counts = df[column].value_counts()
    
    # Calcular proporciones
    total_games = counts.sum()
    proportions = counts / total_games
    
    # Aplicar ambos filtros
    mask = (counts >= min_games) & (proportions >= min_proportion)
    valid_increments = counts[mask].index
    
    return df[df[column].isin(valid_increments)]

def print_frecuency(data):
    # Sacar los valores de mis datos y sus frecuencias
    list_data_incremento = data['increment_code'].nunique()

    print(list_data_incremento)
    incremento = data.groupby('increment_code').count()
    incremento = incremento.sort_values('victory_status')
    print(incremento.iloc[:,0])
    
    frecuencias = incremento.groupby('victory_status').count()
    print(frecuencias.iloc[:,0])

def selection(dataframe):

    # entered_file_name = input("Dime el nombre del archivo CSV: ")

    # file_name = entered_file_name + ".csv"

    # # Carga el conjunto de datos de entrada
    # dataframe = pd.read_csv(file_name)

    # Carga las características o variables predictoras en X (todas las columnas menos la última)
    X = dataframe.iloc[:, :-1]

    # La variable objetivo 'y' (target) es la última columna
    y = dataframe.iloc[:, -1]

    # Obtiene el número de columnas de X
    maximo = X.shape[1]

    # Crea un modelo de bosque aleatorio
    modelo = RandomForestClassifier(random_state=42)


    # Divide el conjunto de datos en entrenamiento (80%) y pruebas (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Inicio de la selección de características
    print("\nTrabajando en la seleccion...")
    best_features = pd.Index([])
    scores=[]
    values=[]
    n_features = 0
    roc = 0

    for i in range(2, maximo+1):
        
        # Crea el selector RFE para seleccionar i características
        selector_caracteristicas = RFE(modelo, n_features_to_select=i, step=1)
    
        # Ajustar el selector al conjunto de datos
        selector_caracteristicas = selector_caracteristicas.fit(X_train, y_train)
        
        # Obtener las características seleccionadas
        selected_features = X.columns[selector_caracteristicas.support_]

        # Crea un nuevo dataframe únicamente con las características seleccionadas
        X_new = dataframe.loc[:, selected_features]

        # Divide el nuevo dataframe en 80% de entrenamiento y 20% de prueba 
        X_ntrain, X_ntest, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

        # Ajusta el modelo 
        modelo.fit(X_ntrain,y_train)

        # Hace predicciones con el modelo ajustado 
        y_pred = modelo.predict(X_ntest)

        # Calcular ROC AUC ('y' reales vs 'y' predichas)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Guarda para graficar
        scores.append(roc_auc)
        values.append(i)
        
        # Guarda el mejor ROC AUC hallado hasta el momento
        if roc_auc > roc:
            roc = roc_auc
            best_features = selected_features
            n_features = i

    # Fin de la selección de características
    

    # Graficar resultados
    title = f"Selección de características usando RFE (CHESS)"
    plt.figure(figsize=(10, 6))
    plt.plot(values, scores, marker='o')
    plt.xlabel('Número de características seleccionadas')
    plt.ylabel('ROC AUC')
    plt.title(title)
    plt.grid(True)
    plt.show()        
        
    # Muestra la lista de características seleccionadas   
    print("\nMejor ROC AUC: ", roc ) 
    message = f"\nMuestra las {n_features} características seleccionadas:\n"
    print(message)
    for feature in best_features:
            print(feature)

    # Comparativa con RFE

    # Crea un dataframe con las mejores características seleccionadas
    X_best = dataframe.loc[:, best_features]

    # Divide el conjunto de datos en 80% de entrenamiento y 20% de prueba 
    X_btrain, X_btest, y_train, y_test = train_test_split(X_best, y, test_size=0.2, random_state=42)

    # Ajusta el modelo
    modelo.fit(X_btrain,y_train)

        # Hace predicciones
    y_pred = modelo.predict(X_btest)

    # Calcular las métricas ('y' reales vs 'y' predichas)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Imprimir las métricas
    print("\nMetricas de rendimiento:\n")
    print("Exactitud (Accuracy):", accuracy)
    print("Precisión (Precision):", precision)
    print("Sensibilidad (Recall):", recall)
    print("Puntuación F1 (F1 Score):", f1)
    print("ROC AUC: ",roc_auc )


    # Despliega la matriz de confusión
    print("Matriz de confusión")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.PuBuGn)
    plt.show()


    # Salva el modelo entrenado para futuras aplicaciones
    model_and_features = {
        'model': modelo,
        'feature_names': best_features
    }

    output_file_name = f"chess_random_forest_model.joblib"
    joblib.dump(model_and_features, output_file_name)

def K_NN(dataframe):

    

    # Obtiene el modelo salvado en previamente

    model_name = "chess_random_forest_model.joblib"

    model_and_features = joblib.load(model_name)

    # Recupera el índice con los nombres de las características seleccionadas

    list_features = list(model_and_features['feature_names'])

    # Muestra los nombres de las características
    print('\nLas características elegidas son:')
    print(list_features)

    # Recupera los datos del dataset
    

    # Obtiene las variables predictoras y objetivo
    X = dataframe[list_features]
    y = dataframe.iloc[:, -1]

    rangos = {}
    for column in X:
        # Filtrar valores que no son cero y encontrar el mínimo
        min_value = X[column][X[column] > 0].min()
        max_value = X[column].max()
        rangos[column] = {'min': min_value, 'max': max_value}


    # Normalizar los valores
    escaler = StandardScaler()
    X = escaler.fit_transform(X)

    # Divide el conjunto de datos en entrenamiento (80%) y pruebas (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Calcular k (mediante una forma de obtener el número de elementos de X)
    k = int(np.sqrt( X.shape[0]));

    if k%2==0: #Hacer que K sea impar 
        k=k+1

    # Crea un modelo de machine learning k-NN 
    k_nn_model = KNeighborsClassifier(n_neighbors=k)

    # Entrenamiento
    k_nn_model.fit(X_train, y_train)

    #Prueba el modelo 
    y_pred = k_nn_model.predict(X_test)

    # Calcular las métricas ('y' reales vs 'y' predichas)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)


    TN, FP, FN, TP = conf_matrix.ravel()


    print("\nResultados de las pruebas:\n")
    print("Muestra de prueba analizadas:", len(X_test))
    print(f"Verdaderos negativos    (TN): {TN}")
    print(f"Falsos positivos        (FP): {FP}")
    print(f"Falsos negativos        (FN): {FN}")
    print(f"Verdaderos positivos    (TP): {TP}")

    # Imprimir las métricas
    print("\nMetricas de rendimiento:\n")
    print("Exactitud (Accuracy)    :", accuracy, "")
    print("Precisión (Precision)   :", precision)
    print("Sensibilidad (Recall)   :", recall)
    print("Puntuación F1 (F1 Score):", f1)
    print("ROC AUC                 :",roc_auc )

    with open("metrics_report_knn.txt", "w") as file:
        file.write("Metricas de rendimiento:\n")
        file.write(f"Exactitud (Accuracy): {accuracy}\n")
        file.write(f"Precisión (Precision): {precision}\n")
        file.write(f"Sensibilidad (Recall): {recall}\n")
        file.write(f"Puntuación F1 (F1 Score): {f1}\n")
        file.write(f"ROC AUC: {roc_auc}\n")
        file.write("\nDetalles adicionales:\n")
        file.write(f"Muestra de prueba analizadas: {len(X_test)}\n")
        file.write(f"Verdaderos negativos (TN): {TN}\n")
        file.write(f"Falsos positivos (FP): {FP}\n")
        file.write(f"Falsos negativos (FN): {FN}\n")
        file.write(f"Verdaderos positivos (TP): {TP}\n")


    # Despliega la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap="BuGn")

    # Aplica análisis de componentes principales
    # para reducir las dimensiones y poder graficar en 2D

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)


    df_pca = pd.DataFrame(data=X_pca, columns=['X_1', 'X_2'])
    df_pca['objetivo'] = y

    # Gráfica de dispersión (Componentes principales)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_pca['X_1'], df_pca['X_2'], c=df_pca['objetivo'], cmap='rainbow')
    plt.colorbar(scatter, label='Target')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Gráfica de dispersión')
    plt.show()


    saved_model_name = "k_nn_model.joblib"
    joblib.dump(k_nn_model, saved_model_name)

    
    
            
    # eFin
    

def Tree(dataframe):
    
    # Obtiene el modelo salvado en previamente
    model_and_features = joblib.load("chess_random_forest_model.joblib")

    # Recupera el índice con los nombres de las características seleccionadas

    list_features = list(model_and_features['feature_names'])

    # Recupera los datos del dataset

    # Muestra los nombres de las características
    print('\n Las características elegidas son:')
    print(list_features)

    # Obtiene las variables predictoras y objetivo
    X = dataframe[list_features]
    y = dataframe['winner']

    # Divide el conjunto de datos en entrenamiento (80%) y pruebas (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #Entrenamiento
    tree_model = DecisionTreeClassifier(criterion='gini',random_state=42)
    tree_model.fit(X_train,y_train)


    #Prueba el modelo 
    y_pred = tree_model.predict(X_test)

    # Calcular las métricas ('y' reales vs 'y' predichas)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Imprimir las métricas
    print("\nMetricas de rendimiento:\n")
    print("Exactitud (Accuracy):", accuracy)
    print("Precisión (Precision):", precision)
    print("Sensibilidad (Recall):", recall)
    print("Puntuación F1 (F1 Score):", f1)
    print("ROC AUC: ",roc_auc )
    TN, FP, FN, TP = conf_matrix.ravel()

    # Despliega la matriz de confusión
    print("Matriz de confusión")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap="BuGn")

    #Visualización

    px = 1/plt.rcParams['figure.dpi']  # Pixel in pulgadas


    # Ajustes de imagen para árbol de breast_cancer
    #fig_size = 2000
    #font_size = 6
    #dots_per_inch = 300


    # Ajustes de imagen para árbol de diabetes
    fig_size = 15000
    font_size = 2
    dots_per_inch = 400

    fig = plt.figure(figsize=(fig_size*px,fig_size*px)) 
    _ = plot_tree(tree_model, feature_names=list_features, class_names=['Negative','Positive'], filled=True, fontsize=font_size, rounded=True)

    # Guarda la imagen del árbol


    with open("tree_metrics_report.txt", "w") as file:
        file.write("Metricas de rendimiento:\n")
        file.write(f"Exactitud (Accuracy): {accuracy}\n")
        file.write(f"Precisión (Precision): {precision}\n")
        file.write(f"Sensibilidad (Recall): {recall}\n")
        file.write(f"Puntuación F1 (F1 Score): {f1}\n")
        file.write(f"ROC AUC: {roc_auc}\n")
        file.write("\nDetalles adicionales:\n")
        file.write(f"Muestra de prueba analizadas: {len(X_test)}\n")
        file.write(f"Verdaderos negativos (TN): {TN}\n")
        file.write(f"Falsos positivos (FP): {FP}\n")
        file.write(f"Falsos negativos (FN): {FN}\n")
        file.write(f"Verdaderos positivos (TP): {TP}\n")

    plt.savefig("arbol_de_decision_diabetes.png", dpi=dots_per_inch, bbox_inches='tight') 
    plt.show()
    saved_model_name = "decision_tree_model.joblib"
    joblib.dump(tree_model, saved_model_name)

def random_values_interactive( dataoriginal,white_rating,black_rating,opening_ply, victory_status,opening_eco, increment_code):
    # Cargar el modelo y características seleccionadas
    model_name = "chess_random_forest_model.joblib"
    model_and_features = joblib.load(model_name)
    list_features = list(model_and_features['feature_names'])
    print('\nLas características elegidas son:')
    print('---------------------------')
    print(list_features)

    # Filtrar las columnas seleccionadas del dataframe
    # X = dataframe[list_features]
    datos_x_original = dataoriginal[['white_rating','black_rating','opening_ply']]
    
    # Identificar las columnas numéricas y sus rangos (min, max)
    print(datos_x_original)
    rangos = {}
    for column in datos_x_original.columns:
        min_value = datos_x_original[column].min()
        max_value = datos_x_original[column].max()
        rangos[column] = {'min': min_value, 'max': max_value}
    
    # Mostrar los rangos detectados para las columnas numéricas
    print("\nRangos detectados para las columnas numéricas:")
    for columna, valores in rangos.items():
        print(f"  {columna}: Mínimo={valores['min']}, Máximo={valores['max']}")

    # Generar valores interactivos
    prod_features = []
    lista_valores = [white_rating, black_rating,opening_ply]
    # Generar valores aleatorios para las columnas numéricas (entre min y max)
    i = 0
    for columna, valores in rangos.items():
        print(f"\nPara la columna '{columna}', ingresa un valor entre {valores['min']} y {valores['max']}:")
        while True:
            try:
                user_input = float(lista_valores[i])
                i = i + 1
                if user_input < valores['min'] or user_input > valores['max']:
                    print(f"El valor debe estar entre {valores['min']} y {valores['max']}. Intenta de nuevo.")
                else:
                    prod_features.append(user_input)
                    break
            except ValueError:
                print("Por favor, ingresa un número válido.")

    # 1. Generar valores para las columnas de 'victory_status' (solo un True)
    victory_status_features = [col for col in list_features if 'victory_status' in col]
    if victory_status_features:
        print("\nSelecciona una de las siguientes opciones para 'victory_status':")
        for i, col in enumerate(victory_status_features, 1):
            print(f"{i}. {col}")
        selected_victory_status_idx = int(victory_status)
        selected_victory_status = victory_status_features[selected_victory_status_idx - 1]
        
        for columna in victory_status_features:
            prod_features.append(1 if columna == selected_victory_status else 0)

    # 2. Generar valores para las columnas de 'opening_eco' (incluye opción de no seleccionar ninguna)
    opening_eco_features = [col for col in list_features if 'opening_eco' in col]
    if opening_eco_features:
        print("\nSelecciona una de las siguientes opciones para 'opening_eco':")
        for i, col in enumerate(opening_eco_features, 1):
            print(f"{i}. {col}")
        print(f"{len(opening_eco_features) + 1}. No seleccionar ninguna")
        selected_opening_idx = int(opening_eco)
        
        if selected_opening_idx <= len(opening_eco_features):
            selected_opening = opening_eco_features[selected_opening_idx - 1]
        else:
            selected_opening = None
        
        for columna in opening_eco_features:
            prod_features.append(1 if columna == selected_opening else 0)

    # 3. Generar valores para las columnas de 'increment_code' (incluye opción de no seleccionar ninguna)
    increment_code_features = [col for col in list_features if 'increment_code' in col]
    if increment_code_features:
        print("\nSelecciona una de las siguientes opciones para 'increment_code':")
        for i, col in enumerate(increment_code_features, 1):
            print(f"{i}. {col}")
        print(f"{len(increment_code_features) + 1}. No seleccionar ninguna")
        selected_increment_idx = int(increment_code)
        
        if selected_increment_idx <= len(increment_code_features):
            selected_increment = increment_code_features[selected_increment_idx - 1]
        else:
            selected_increment = None
        
        for columna in increment_code_features:
            prod_features.append(1 if columna == selected_increment else 0)

    # Crear un dataframe con los datos generados
    df_prod = pd.DataFrame([prod_features], columns=list_features)
    print("\nDatos generados:")
    print(df_prod)
    discrete_columns = ['white_rating', 'black_rating', 'opening_ply']
    scaler = MinMaxScaler()
    scaler.fit(datos_x_original[discrete_columns])  # Entrenar el escalador con los datos originales
    df_prod[discrete_columns] = scaler.transform(df_prod[discrete_columns])  # Escalar los datos generados

    # Crear un dataframe con los datos normalizados
    df_prod_normalized = pd.DataFrame(df_prod, columns=list_features)
    print(df_prod_normalized)
    return df_prod_normalized



