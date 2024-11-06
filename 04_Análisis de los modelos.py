import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

###Importar imagenes
x_train = joblib.load('Salidas/x_train.pkl')
y_train = joblib.load('Salidas/y_train.pkl')
x_test = joblib.load('Salidas/x_test.pkl')
y_test = joblib.load('Salidas/y_test.pkl')

#### Escalar ######
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 

x_train /=255 
x_test /=255


##### cargar modelo h5  ######

##modelo=tf.keras.models.load_model('best_model_cnn.h5')
modelo = XGBClassifier()

modelo.load_model('best_xgb.model')


####desempeño en evaluación para grupo 1 (tienen cancer de piel) #######
prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold


threshold_neu=0.508

pred_test=(modelo.predict(x_test)>=threshold_neu).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()



### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_neu).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()


########### ##############################################################
####desempeño en evaluación para grupo 1 (No tienen neumonía) #######
########### ##############################################################

prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold


threshold_no_neu=0.5015

pred_test=(modelo.predict(x_test)>=threshold_no_neu).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()



### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_no_neu).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()



####### clasificación final ################

prob=modelo.predict(x_test)

clas = ['Malignant' if prob > 0.495 else 'Benign' for prob in prob]

# Contar las clases únicas
clases, count = np.unique(clas, return_counts=True)

# Calcular el porcentaje de cada clase
count * 100 / np.sum(count)

#### Para XgBoost #####

# Cargar el modelo XGBoost
modelo = XGBClassifier()
modelo.load_model('best_xgb.model')

# Definir función para evaluación y visualización de resultados
def evaluar_modelo(modelo, x_data, y_data, threshold, dataset_name="Test"):
    # Predicción de probabilidades
    prob = modelo.predict_proba(x_data)[:, 1]  # Probabilidad para clase 1 (Malignant)
    
    # Histograma de probabilidades
    sns.histplot(prob, legend=False)
    plt.title(f"Probabilidades - {dataset_name}")
    plt.show()

    # Aplicar umbral para predicciones
    pred = (prob >= threshold).astype('int')
    
    # Mostrar el reporte de clasificación
    print(f"Classification Report - {dataset_name} Dataset")
    print(metrics.classification_report(y_data, pred))
    
    # Matriz de confusión
    cm = metrics.confusion_matrix(y_data, pred, labels=[1, 0])
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Malignant', 'Benign'])
    disp.plot()
    plt.show()

# Evaluación en datos de prueba y entrenamiento con diferentes thresholds
threshold_malignant = 0.508
threshold_benign = 0.5015

print("Evaluación en conjunto de prueba (Malignant Threshold)")
evaluar_modelo(modelo, x_test, y_test, threshold_malignant, dataset_name="Test - Malignant")

print("Evaluación en conjunto de entrenamiento (Malignant Threshold)")
evaluar_modelo(modelo, x_train, y_train, threshold_malignant, dataset_name="Train - Malignant")

print("Evaluación en conjunto de prueba (Benign Threshold)")
evaluar_modelo(modelo, x_test, y_test, threshold_benign, dataset_name="Test - Benign")

print("Evaluación en conjunto de entrenamiento (Benign Threshold)")
evaluar_modelo(modelo, x_train, y_train, threshold_benign, dataset_name="Train - Benign")

# Clasificación final basada en probabilidad
prob_test = modelo.predict_proba(x_test)[:, 1]
clas_final = ['Malignant' if prob > 0.495 else 'Benign' for prob in prob_test]

# Contar las clases únicas y calcular el porcentaje de cada clase
clases, count = np.unique(clas_final, return_counts=True)
porcentajes = count * 100 / np.sum(count)
print("Clasificación Final:")
for clase, porcentaje in zip(clases, porcentajes):
    print(f"{clase}: {porcentaje:.2f}%")