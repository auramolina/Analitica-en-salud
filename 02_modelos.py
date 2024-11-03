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

### cargar bases_procesadas ####
x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')


#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=255 ### escalaro para que quede entre 0 y 1
x_test /=255

###### verificar tamaños

x_train.shape
x_test.shape

np.product(x_train[1].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)


############################################################
############## Probar modelos de tradicionales #############
############################################################

####################### RandomForest #######################

x_train2 = x_train.reshape(2637,30000)
x_test2 = x_test.reshape(660, 30000)
x_train2.shape
x_test2.shape

rf = RandomForestClassifier()
rf.fit(x_train2, y_train)

print('------------------TRAIN RANDOM FOREST---------------------------')
pred_train = rf.predict_proba(x_train2)[:, 1]
pred_train = (pred_train > 0.7).astype(int)
print(metrics.classification_report(y_train, pred_train))
metrics.roc_auc_score(y_train, pred_train)

print('------------------TEST RANDOM FOREST---------------------------')
pred_test = rf.predict_proba(x_test2)[:, 1]
pred_test = (pred_test > 0.7).astype(int)
print(metrics.classification_report(y_test, pred_test))
metrics.roc_auc_score(y_test, pred_test)

# matriz de confusión test
pred_test = (rf.predict(x_test2) > 0.70).astype('int')
cm = metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp = metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()

print(metrics.classification_report(y_test, pred_test))

###################### Decsion tree #######################

clf_dt = tree.DecisionTreeClassifier()
clf = clf_dt.fit(x_train2, y_train)

print('-----------------TRAIN DECISION TREE---------------------------')
pred_train = clf_dt.predict_proba(x_train2)[:, 1]
pred_train = (pred_train > 0.7).astype(int)
print(metrics.classification_report(y_train, pred_train))
metrics.roc_auc_score(y_train, pred_train)


print('------------------TEST DECISION TREE---------------------------')
pred_test = clf_dt.predict_proba(x_test2)[:, 1]
pred_test = (pred_test > 0.7).astype(int)
print(metrics.classification_report(y_test, pred_test))
metrics.roc_auc_score(y_test, pred_test)

# matriz de confusión test
pred_test = (clf_dt.predict(x_test2) > 0.70).astype('int')
cm = metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp = metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()

print(metrics.classification_report(y_test, pred_test))

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
fc_model.fit(x_train, y_train, batch_size=100, epochs=30, validation_data=(x_test, y_test))

# Evaluar el modelo
test_loss, test_acc, test_auc, test_recall, test_precision = fc_model.evaluate(x_test, y_test, verbose=2)
print("Test auc:", test_auc)
x_test.shape

# matriz de confusión test
pred_test = (fc_model.predict(x_test) > 0.50).astype('int')
cm = metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp = metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()

print(metrics.classification_report(y_test, pred_test))

fc_model.save("best_model.h5")

