# Importa las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
import seaborn as sns
import numpy as np

data = pd.read_csv('features.csv')

print(data.info())

# Crea un histograma objetivo v subjetivo
plt.figure(figsize=(8, 6))
data['Label'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribución de clases (Label)')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()

# Dividir los datos en características (X) y etiquetas (y)
X = data.drop(['Label'], axis=1)
y = data['Label']

# Eliminar las columnas no numéricas
non_numeric_columns = X.select_dtypes(exclude=['number']).columns
X = X.drop(non_numeric_columns, axis=1)

# Realizar un stratified shuffle para dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Escala las características para SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GridSearchCV para SVM
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}
svm_grid = GridSearchCV(SVC(random_state=42), svm_param_grid, refit=True, verbose=0)
svm_grid.fit(X_train_scaled, y_train)

# GridSearchCV para Random Forest
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, refit=True, verbose=0)
rf_grid.fit(X_train, y_train)

# GridSearchCV para Extra Trees
et_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None]}
et_grid = GridSearchCV(ExtraTreesClassifier(random_state=42), et_param_grid, refit=True, verbose=0)
et_grid.fit(X_train, y_train)

# Obteniendo predicciones con los modelos ajustados
svm_grid_predictions = svm_grid.predict(X_test_scaled)
rf_grid_predictions = rf_grid.predict(X_test)
et_grid_predictions = et_grid.predict(X_test)

# matriz de confusión, precisión, recall y F1-score para cada clasificador
def evaluate_classifier(predictions, classifier_name):
    cm = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    print(f'Clasificador: {classifier_name}')
    print(f'Matriz de Confusión:\n{cm}')
    print(f'Precisión: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}\n')

evaluate_classifier(svm_grid_predictions, 'SVM')
evaluate_classifier(rf_grid_predictions, 'Random Forest')
evaluate_classifier(et_grid_predictions, 'Extra Trees')

# valores binarios
y_test_binary = y_test.map({'objective': 0, 'subjective': 1})

# curva de precisión-recall para todos los clasificadores
def plot_precision_recall_curve(clf, X_test, y_test, label):
    if hasattr(clf, "decision_function"):
        decision_function = clf.decision_function(X_test)
    elif hasattr(clf, "predict_proba"):
        decision_function = clf.predict_proba(X_test)[:, 1]
    else:
        raise AttributeError("El clasificador no tiene un método para obtener la decisión")

    precision, recall, _ = precision_recall_curve(y_test, decision_function)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva de Precisión-Recall ({label})')
    plt.legend()
    plt.show()

plot_precision_recall_curve(svm_grid, X_test_scaled, y_test_binary, 'SVM (GridSearchCV)')
plot_precision_recall_curve(rf_grid, X_test, y_test_binary, 'Random Forest (GridSearchCV)')
plot_precision_recall_curve(et_grid, X_test, y_test_binary, 'Extra Trees (GridSearchCV)')

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import auc


#roc curve
def plot_roc_curve(clf, X_test, y_test, label):
    if hasattr(clf, "decision_function"):
        decision_function = clf.decision_function(X_test)
    elif hasattr(clf, "predict_proba"):
        decision_function = clf.predict_proba(X_test)[:, 1]
    else:
        raise AttributeError("El clasificador no tiene un método para obtener la decisión")

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    
    fpr, tpr, _ = roc_curve(y_test_bin, decision_function)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({label})')
    plt.legend(loc="best")
    plt.show()

plot_roc_curve(svm_grid, X_test_scaled, y_test_binary, 'SVM (GridSearchCV)')
plot_roc_curve(rf_grid, X_test, y_test_binary, 'Random Forest (GridSearchCV)')
plot_roc_curve(et_grid, X_test, y_test_binary, 'Extra Trees (GridSearchCV)')

def plot_confusion_matrix(model, X_test, y_train):
    y_train_pred =cross_val_predict(model, X_test, y_test, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)

plot_confusion_matrix(svm_grid, X_test_scaled, y_test)
plot_confusion_matrix(rf_grid, X_test, y_test)
plot_confusion_matrix(et_grid, X_test, y_test)

#mapa de calor
def plot_confusion_matrix_map(model, X_test, y_train):
    y_train_pred =cross_val_predict(model, X_test, y_test, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    plt.matshow(conf_mx, cmap=plt.cm.gray)

plot_confusion_matrix_map(svm_grid, X_test_scaled, y_test)
plot_confusion_matrix_map(rf_grid, X_test, y_test)
plot_confusion_matrix_map(et_grid, X_test, y_test)

import pickle
filename='trained_model.sav'
pickle.dump(svm_grid, open(filename, 'wb'))
