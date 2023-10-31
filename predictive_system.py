import numpy as np
import pandas as pd
import pickle

# Cargar el modelo
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Leer tu nuevo conjunto de datos
new_data = pd.read_csv('features.csv')

features = new_data.drop(['TextID', 'URL','Label'], axis=1)

# Realizar la predicción con el modelo cargado
predictions = loaded_model.predict(features)
print(predictions)
