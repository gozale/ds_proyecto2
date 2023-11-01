import pickle
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

loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

input_data = (1577,97,51,38,131,5,35,141,128,5,8,0,26,363,8,0,96,1,15,49,29,72,1,1,5,0,37,0,19,34,31,21,31,71,0,6,0,3,0,52,0,3,2,130,35,0,3,0,5,2,46,15,34,19,71,31,1,1,11)

input_data_as_numpy_array = np.asarray(input_data) 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

