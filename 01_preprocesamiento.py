#librerías
import os
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


############################################################
####### Carga de imagenes y dividir en train/test ##########
############################################################

# Parámetros
width = 100  # tamaño para reescalar la imagen
num_classes = 2  # cantidad de clases
datapath = 'Data/'  # ruta de la carpeta principal con subcarpetas 'no' y 'yes'


# Cargar imágenes y etiquetas
x_data, y_data = fn.load_images_and_labels(datapath, width)

# Convertir a arrays numpy
x_data = np.array(x_data)
y_data = np.array(y_data)

# Dividir en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Verificar las formas de los datos
x_train.shape
y_train.shape
x_test.shape
y_test.shape

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "Salidas/x_train.pkl")
joblib.dump(y_train, "Salidas/y_train.pkl")
joblib.dump(x_test, "Salidas/x_test.pkl")
joblib.dump(y_test, "Salidas/y_test.pkl")