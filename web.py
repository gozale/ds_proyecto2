import streamlit as st
import pandas as pd
import pickle

# Cargar el modelo entrenado
filename = 'trained_model.sav'  # Asegúrate de tener el nombre correcto de tu archivo de modelo
loaded_model = pickle.load(open(filename, 'rb'))

st.title('Predictor de Objetivo o Subjetivo')

st.write('Este predictor clasifica si un documento es objetivo o subjetivo.')

# Recopilar la entrada del usuario
totalWordsCount = st.number_input('Total de palabras', min_value=0)
semanticobjscore = st.number_input('Puntuación semántica objetivo')
semanticsubjscore = st.number_input('Puntuación semántica subjetiva')
# Agrega otros parámetros numéricos según sea necesario para tu modelo

# Hacer la predicción
if st.button('Predecir'):
    input_data = [[totalWordsCount, semanticobjscore, semanticsubjscore]]  # Agrega otros parámetros aquí
    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        st.write('El documento es objetivo.')
    else:
        st.write('El documento es subjetivo.')
