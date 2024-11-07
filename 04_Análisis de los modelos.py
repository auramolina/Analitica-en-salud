import numpy as np
import joblib
import tensorflow as tf
from sklearn import metrics 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.decomposition import PCA 


### Importar imagenes
x_train = joblib.load('Salidas/x_train.pkl')
y_train = joblib.load('Salidas/y_train.pkl')
x_test = joblib.load('Salidas/x_test.pkl')
y_test = joblib.load('Salidas/y_test.pkl')

### Escalar
x_train=x_train.astype('float32')
x_test=x_test.astype('float32') 

x_train /=255 
x_test /=255


############################################################
########################## XGBOOST #########################
############################################################

# Aplicar PCA para reducir a 100 dimensiones
pca = PCA(n_components=100)
x_train_pca = pca.fit_transform(x_train.reshape(len(x_train), -1))
x_test_pca = pca.transform(x_test.reshape(len(x_test), -1))

# Cargar el modelo XGBoost
modeloxgb = XGBClassifier()
modeloxgb.load_model('xgb_model.model')

# Umbrales para clasificar como 'Malignant' o 'Benign'
threshold_malignant = 0.7
threshold_benign = 0.2

# Función para evaluar y mostrar métricas
def evaluar_conjunto(x_data, y_data, conjunto_nombre):
    # Predicción de probabilidades
    prob = modeloxgb.predict_proba(x_data)[:, 1]

    # Visualización de distribución de probabilidades
    sns.histplot(prob, bins=20, legend=False)
    plt.title(f"Distribución de probabilidades en el conjunto de {conjunto_nombre}")
    plt.xlabel("Probabilidad de clase Malignant")
    plt.show()

    # Clasificación con el umbral 'Malignant'
    pred = (prob >= threshold_malignant).astype('int')
    print(f"Evaluación en el conjunto de {conjunto_nombre}:")
    print(metrics.classification_report(y_data, pred))

    # Matriz de confusión
    cm = metrics.confusion_matrix(y_data, pred, labels=[1, 0])
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Malignant', 'Benign'])
    disp.plot()
    plt.title(f"Matriz de Confusión - {conjunto_nombre}")
    plt.show()

    # Clasificación en tres categorías: Malignant, Benign, y Uncertain
    clasificacion = [
        'Malignant' if p >= threshold_malignant 
        else 'Benign' if p <= threshold_benign 
        else 'Uncertain' 
        for p in prob
    ]

    # Distribución final de clases
    clases, count = np.unique(clasificacion, return_counts=True)
    print(f"Distribución final de clases en el conjunto de {conjunto_nombre}:")
    for clase, cnt in zip(clases, count):
        print(f"{clase}: {cnt * 100 / np.sum(count):.2f}%")
    print("\n")

# Evaluar ambos conjuntos
evaluar_conjunto(x_train_pca, y_train, "entrenamiento")
evaluar_conjunto(x_test_pca, y_test, "prueba")