#librerías
import os
from os import listdir
import cv2 
import numpy as np 
import funciones as fn
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report
import joblib ### para descargar array

#tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


#############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

img1 = cv2.imread('Data/train/benign/3.jpg')
img2 = cv2.imread('Data/train/malignant/2.jpg')


############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

#### Cancer de piel: benigno ######
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img1)
axs[0].set_title(f'Benign. Size : {img1.shape[0]}x{img1.shape[1]}')

img1_r = cv2.resize(img1 ,(100,100))
axs[1].imshow(img1_r)
axs[1].set_title('Benign 100x100')

plt.show()

#### Cancer de piel: maligno ######
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img2)
axs[0].set_title(f'Malignant. Size : {img2.shape[0]}x{img2.shape[1]}')

img1_r = cv2.resize(img2 ,(100,100))
axs[1].imshow(img1_r)
axs[1].set_title('Malignant 100x100')

plt.show()

############################################################
####### Carga de imagenes y dividir en train/test ##########
############################################################

width = 100 #tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'Data/train/'
testpath = 'Data/test/'

x_train, y_train = fn.img2data(trainpath) #Run in train
x_test, y_test = fn.img2data(testpath) #Run in test

# Verificar las formas de los datos
x_train.shape
x_test.shape


np.prod(x_train[1].shape)
y_train.shape


x_test.shape
y_test.shape


####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "Salidas/x_train.pkl")
joblib.dump(y_train, "Salidas/y_train.pkl")
joblib.dump(x_test, "Salidas/x_test.pkl")
joblib.dump(y_test, "Salidas/y_test.pkl")