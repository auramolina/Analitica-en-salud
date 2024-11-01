import numpy as np
import joblib
import cv2  # Para procesar imágenes
import tensorflow as tf
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA  # Para reducir dimensionalidad
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import pandas as pd

# Cargar datos preprocesados
x_train = joblib.load('salidas/x_train.pkl')
y_train = joblib.load('salidas/y_train.pkl')
x_test = joblib.load('salidas/x_test.pkl')
y_test = joblib.load('salidas/y_test.pkl')

# Reducción de dimensionalidad para modelos de Random Forest y Decision Tree
pca = PCA(n_components=100)  
x_train_reduced = pca.fit_transform(x_train.reshape(len(x_train), -1))
x_test_reduced = pca.transform(x_test.reshape(len(x_test), -1))

# Optimización y ajuste del modelo Random Forest
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

# Evaluación de Random Forest en el conjunto de prueba
pred_test_rf = best_rf.predict(x_test_reduced)
print(metrics.classification_report(y_test, pred_test_rf))
print("Random Forest AUC:", metrics.roc_auc_score(y_test, pred_test_rf))

# Optimización y ajuste del modelo Decision Tree
dt = DecisionTreeClassifier()
param_grid_dt = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search_dt = RandomizedSearchCV(dt, param_distributions=param_grid_dt, n_iter=10, cv=3, scoring='roc_auc')
random_search_dt.fit(x_train_reduced, y_train)
best_dt = random_search_dt.best_estimator_

# Evaluación de Decision Tree en el conjunto de prueba
pred_test_dt = best_dt.predict(x_test_reduced)
print(metrics.classification_report(y_test, pred_test_dt))
print("Decision Tree AUC:", metrics.roc_auc_score(y_test, pred_test_dt))

# Aumentación de datos y entrenamiento de redes neuronales
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

# Red neuronal con regularización y Dropout
reg_strength = 0.001
dropout_rate = 0.3

# Mantener las imágenes en su forma original para la red neuronal
x_train_nn = x_train / 255.0  # Normalización de las imágenes
x_test_nn = x_test / 255.0

fc_model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Flatten(input_shape=(100, 100, 3)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilación del modelo
fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'Recall', 'Precision'])

# Entrenamiento del modelo de red neuronal
fc_model.fit(x_train_nn, y_train, batch_size=64, epochs=10, validation_data=(x_test_nn, y_test))

# Evaluación del modelo en el conjunto de prueba
test_loss, test_auc, test_recall, test_precision = fc_model.evaluate(x_test, y_test, verbose=2)
print("Test AUC:", test_auc)

# Guardar el modelo con mejores métricas
fc_model.save('mejor_modelo.h5')
joblib.dump(best_rf, 'Salidas/mejor_modelo_rf.pkl')
joblib.dump(best_dt, 'Salidas/mejor_modelo_dt.pkl')
