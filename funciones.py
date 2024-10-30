import numpy as np
import os
#from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
#from os.path import join ### para unir ruta con archivo 
import cv2 ### para leer imagenes jpg
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Función para cargar imágenes y etiquetas
def load_images_and_labels(datapath, width):
    x_data = []
    y_data = []
    labels = {'no': 0, 'yes': 1}  # etiquetas para las clases

    for label, class_num in labels.items():
        class_path = os.path.join(datapath, label)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = load_img(img_path, target_size=(width, width))
            img_array = img_to_array(img)
            x_data.append(img_array)
            y_data.append(class_num)

    return x_data, y_data


def img2data(path, width=100):
    
    rawImgs = []   #### una lista con el array que representa cada imágen
    labels = [] ### el label de cada imágen
    
    list_labels = [path+f for f in listdir(path)] ### crea una lista de los archivos en la ruta (Normal /Pneumonia)

    for imagePath in list_labels: ### recorre cada carpeta de la ruta ingresada
        
        files_list=listdir(imagePath) ### crea una lista con todos los archivos
        for item in tqdm(files_list): ### le pone contador a la lista: tqdm
            file = join(imagePath, item) ## crea ruta del archivo
            if file[-1] =='g': ### verificar que se imágen extensión jpg o jpeg
                img = cv2.imread(file) ### cargar archivo
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) ### invierte el orden de los colores en el array para usar el más estándar RGB
                img = cv2.resize(img ,(width,width)) ### cambia resolución de imágnenes
                rawImgs.append(img) ###adiciona imágen al array final
                l = imagePath.split('/')[2] ### identificar en qué carpeta está
                if l == 'NORMAL':  ### verificar en qué carpeta está para asignar el label
                    labels.append([0])
                elif l == 'TUMOR':
                    labels.append([1])
    return rawImgs, labels, files_list