## Proyecto de Clasificación Objetivo o Subjetivo

### Descripción del proyecto
Este proyecto consiste en un sistema de clasificación que determina si un documento es objetivo o subjetivo. Utiliza modelos de aprendizaje automático entrenados con características específicas extraídas de textos. El proyecto contiene varios archivos Python que realizan diversas tareas, desde el preprocesamiento de datos hasta la predicción en base a un modelo entrenado.

### Estructura de archivos
- **main.py**: Contiene el código principal que incluye la importación de bibliotecas, carga de datos, entrenamiento de modelos de aprendizaje automático (SVM, Random Forest, Extra Trees), evaluación de modelos y visualización de métricas.
- **predictive_system.py**: Implementa la predicción utilizando el modelo entrenado previamente. Se carga el modelo y realiza predicciones sobre nuevos datos proporcionados en forma de un conjunto de características.
- **web.py**: Es una aplicación web desarrollada con Streamlit que interactúa con el modelo entrenado. Permite a los usuarios ingresar valores específicos para las características del texto y obtener una predicción en tiempo real sobre si el documento es objetivo o subjetivo.

### Archivos y Funcionalidades
1. **main.py**:
   - Importa bibliotecas y preprocesa datos.
   - Entrena modelos de SVM, Random Forest y Extra Trees mediante GridSearchCV.
   - Evalúa los modelos utilizando métricas como la matriz de confusión, precisión, recall y F1-score.
   - Genera curvas de precisión-recall y curvas ROC para cada clasificador.
   - Guarda el modelo SVM entrenado en un archivo.

2. **predictive_system.py**:
   - Carga el modelo SVM previamente entrenado.
   - Permite ingresar un conjunto de características para realizar una predicción sobre si el documento es objetivo o subjetivo.

3. **web.py**:
   - Implementa una aplicación web interactiva con Streamlit.
   - Permite a los usuarios ingresar valores para diferentes características del texto.
   - Realiza una predicción en tiempo real sobre si el documento es objetivo o subjetivo.

### Ejecución
Para ejecutar el proyecto, se deben seguir estos pasos:

1. Ejecutar `main.py` para entrenar los modelos y guardar el modelo SVM entrenado.
2. Utilizar `predictive_system.py` para realizar predicciones con el modelo guardado.
3. Acceder a la aplicación web utilizando `web.py` mediante el enlace proporcionado(streamlit): [Objetivo o Subjetivo Predictor](https://objorsubj.streamlit.app/)
