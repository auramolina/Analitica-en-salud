import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

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


##### cargar modelo  ######

modelo=tf.keras.models.load_model('best_model_cnn.h5')



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

clas=['Malignant' if prob >0.508 else 'Benign' if prob <0.5015 else "No ident" for prob in prob]

clases, count =np.unique(clas, return_counts=True)

count*100/np.sum(count)