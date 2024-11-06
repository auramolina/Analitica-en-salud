import numpy as np
import joblib 
# Paquetes para NN 
import tensorflow as tf
from sklearn import metrics # para analizar modelo
from sklearn.ensemble import RandomForestClassifier  # para analizar modelo
import pandas as pd
from sklearn import tree
import cv2 # para leer imagenes jpeg
### pip install opencv-python
from matplotlib import pyplot as plt #
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA 

### cargar bases_procesadas ####
x_train = joblib.load('Salidas\\x_train.pkl')
y_train = joblib.load('Salidas\\y_train.pkl')
x_test = joblib.load('Salidas\\x_test.pkl')
y_test = joblib.load('Salidas\\y_test.pkl')

### Escalar 
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=255 ### escalar para que quede entre 0 y 1
x_test /=255

### verificar tamaños
x_train.shape
x_test.shape

np.product(x_train[1].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)


############################################################
############## Probar modelos de tradicionales #############
############################################################

####################### RandomForest #######################

# x_train2 = x_train.reshape(2637,30000)
# x_test2 = x_test.reshape(660, 30000)
# x_train2.shape
# x_test2.shape
# rf = RandomForestClassifier()
# rf.fit(x_train2, y_train)

pca = PCA(n_components=100)
x_train_reduced = pca.fit_transform(x_train.reshape(len(x_train), -1))
x_test_reduced = pca.transform(x_test.reshape(len(x_test), -1))

# Optimización del modelo Random Forest
rf = RandomForestClassifier()
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_grid_rf, n_iter=10, cv=3, scoring='roc_auc')
random_search_rf.fit(x_train_reduced, y_train)
best_rf = random_search_rf.best_estimator_

# Evaluación en el conjunto de entrenamiento
print('------------------TRAIN RANDOM FOREST---------------------------')
pred_train_proba = best_rf.predict_proba(x_train_reduced)[:, 1]
pred_train = (pred_train_proba > 0.5).astype(int)
print(classification_report(y_train, pred_train))
train_auc = roc_auc_score(y_train, pred_train)
print("AUC - Train Random Forest:", train_auc)

# Evaluación en el conjunto de prueba
print('------------------TEST RANDOM FOREST---------------------------')
pred_test_proba = best_rf.predict_proba(x_test_reduced)[:, 1]
pred_test = (pred_test_proba > 0.5).astype(int)
print(classification_report(y_test, pred_test))
test_auc = roc_auc_score(y_test, pred_test)
print("AUC - Test Random Forest:", test_auc)

#Matriz de confusión para el conjunto de prueba
cm_rf = confusion_matrix(y_test, pred_test, labels=[1, 0])
disp = ConfusionMatrixDisplay(cm_rf, display_labels=['Malignant', 'Benign'])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Verdad Real")
plt.show()

####################### XGBoosting ########################

pca = PCA(n_components=100)  
x_train_reduced = pca.fit_transform(x_train.reshape(len(x_train), -1))
x_test_reduced = pca.transform(x_test.reshape(len(x_test), -1))

param_grid_xgb = {
    'n_estimators': [100, 200, 300],        # Número de árboles
    'max_depth': [3, 5, 10],                # Profundidad de cada árbol
    'learning_rate': [0.01, 0.1, 0.2],      # Tasa de aprendizaje
    'subsample': [0.6, 0.8, 1.0],           # Porcentaje de muestras para entrenar cada árbol
    'colsample_bytree': [0.6, 0.8, 1.0]     # Porcentaje de características usadas en cada árbol
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Configuración inicial de XGBoost

# RandomizedSearchCV para ajustar los parámetros
random_search_xgb = RandomizedSearchCV(
    xgb, param_distributions=param_grid_xgb, n_iter=10, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
)
random_search_xgb.fit(x_train_reduced, y_train)  # Entrenamiento
best_xgb = random_search_xgb.best_estimator_     # Modelo con mejores parámetros

print('------------------TRAIN XGBOOST---------------------------')
# Predicción y ajuste de umbral en el conjunto de entrenamiento
pred_train_proba = best_xgb.predict_proba(x_train_reduced)[:, 1]
pred_train = (pred_train_proba > 0.5).astype(int)

# Reporte de métricas en el conjunto de entrenamiento
print(classification_report(y_train, pred_train))
train_auc = roc_auc_score(y_train, pred_train)
print("AUC - Train XGBoost:", train_auc)

print('------------------TEST XGBOOST---------------------------')
# Predicción y ajuste de umbral en el conjunto de prueba
pred_test_proba = best_xgb.predict_proba(x_test_reduced)[:, 1]
pred_test = (pred_test_proba > 0.5).astype(int)

# Reporte de métricas en el conjunto de prueba
print(classification_report(y_test, pred_test))
test_auc = roc_auc_score(y_test, pred_test)
print("AUC - Test XGBoost:", test_auc)

# Matriz de confusión para el conjunto de prueba
cm_xgb = confusion_matrix(y_test, pred_test, labels=[1, 0])
disp = ConfusionMatrixDisplay(cm_xgb, display_labels=['Malignant', 'Benign'])
disp.plot(cmap="Oranges")
plt.title("Matriz de Confusión - XGBoost")
plt.xlabel("Predicción")
plt.ylabel("Verdad Real")
plt.show()


############################################################
############ Probar modelos de redes neuronales ############
############################################################

fc_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# configura el optimizador y la función para optimizar
fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','AUC', 'Recall', 'Precision'])

# Entrenar el modelo usando el optimizador y arquitectura definidas
fc_model.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_test, y_test))

# Evaluar el modelo
test_loss, test_acc, test_auc, test_recall, test_precision = fc_model.evaluate(x_test, y_test, verbose=2)
# print("Test auc:", test_auc)
print('----------------- TEST NN ---------------------------')
print(classification_report(y_test, pred_test))
print("Test auc:", test_auc)

# matriz de confusión test
pred_test = (fc_model.predict(x_test) > 0.50).astype('int')
cm = metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp = metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()
print(metrics.classification_report(y_test, pred_test))


##### Exportar el mejor modelo ######
best_xgb.save_model("best_xgb.model")

