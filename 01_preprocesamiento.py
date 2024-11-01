#librerías
import os
from os import listdir
import cv2 
import numpy as np 
import funciones as fn
import matplotlib.pyplot as plt 
import joblib ### para descargar array

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

#width = 100 #tamaño para reescalar imágen
#num_classes = 2 #clases variable respuesta
trainpath = 'Data/train/'
testpath = 'Data/test/'

x_test, y_test, names_train = fn.import_data(testpath)
x_train, y_train, names_train = fn.import_data(trainpath) #Run in train

## Convertir en array
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Verificar las formas de los datos
print('Dimensiones de x_train:', x_train.shape)
print('Dimensiones de y_train:', y_train.shape)
print('Dimensiones de x_test:', x_test.shape)
print('Dimensiones de y_test:', y_test.shape)



####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "Salidas\\x_train.pkl")
joblib.dump(y_train, "Salidas\\y_train.pkl")
joblib.dump(x_test, "Salidas\\x_test.pkl")
joblib.dump(y_test, "Salidas\\y_test.pkl")