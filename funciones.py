import numpy as np

from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
from os.path import join ### para unir ruta con archivo 
import cv2

def img2data(path, width=100):
    rawImgs = []  # Lista con el array que representa cada imagen
    labels = []   # El label de cada imagen
    
    list_labels = [join(path, f) for f in listdir(path)]  # Construir rutas a las carpetas

    for imagePath in list_labels:  # Recorre cada carpeta
        files_list = listdir(imagePath)  # Lista de archivos en la carpeta
        for item in tqdm(files_list):  # Progreso con tqdm
            file = join(imagePath, item)  # Ruta del archivo
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Verificar extensiones válidas
                img = cv2.imread(file)  # Cargar archivo
                if img is not None:  # Verificar si la imagen se cargó correctamente
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
                    img = cv2.resize(img, (width, width))  # Cambiar resolución
                    rawImgs.append(img)  # Añadir imagen al array final
                    l = imagePath.split('/')[-1]  # Identificar en qué carpeta está
                    if l == 'BENIGN':  # Etiquetar según la carpeta
                        labels.append([0])
                    elif l == 'MALIGNANT':
                        labels.append([1])

    return rawImgs, labels