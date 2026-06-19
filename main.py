import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,  LabelEncoder
import Funciones as f
import os
"""
    PROYECTO FINAL
    FUNDAMENTOS DE INTELIGENCIA ARTIFICIAL

    Participantes:
        Bustillos Cruz Jonatan
        Martínes Contreras Leonardo
        Salazar Bravo Alejandro Román
        Velazquez Matus Álvaro Alexander

    Profesor:
        Hernández Cruz Macario

    
"""
class datos_clase(object):
    def __init__(self,datos,datos_originales):
        self.data = datos
        self.datos_originales = datos_originales

    def devolver_valores(self):
        return self.data, self.datos_clase
    


def hacer_datos():
    data = f.get_data('games.csv')
    # Quitamos los valores atípicos del incrment code, deben 
    # Tener un porcentaje o un número mínimo
    data = f.filter(data, 'increment_code', 0.02)
    data = f.filter(data,'opening_eco', 0.02)
    data_2 = data.copy()
    
    return data,data_2





# Para obtener las características necesarioas
data = f.get_data('games.csv')
# Quitamos los valores atípicos del incrment code, deben 
# Tener un porcentaje o un número mínimo
data = f.filter(data, 'increment_code', 0.02)
data = f.filter(data,'opening_eco', 0.02)
data1,data2 = hacer_datos()
# f.print_frecuency(data)
#  Codificamos según los valores
data = f.code(data)
# Balanceamos la función objetivo
data = f.balance(data)

# Seleccionamos las mejores características por medio de bosques aleatorios.
if not os.path.exists('chess_random_forest_model.joblib'):
    f.selection(data)
    # Las comparamos con las obtenidas con RFE
# f.K_NN(data)
# f.Tree(data)


