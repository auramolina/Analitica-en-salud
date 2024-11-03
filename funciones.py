import numpy as np

from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
from os.path import join ### para unir ruta con archivo 
import cv2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def import_data(path, width = 100):
    
    rawImgs = []   #### una lista con el array que representa cada imágen
    labels = [] ### el label de cada imágen
    names = []
    
    list_labels = [path + f for f in listdir(path)] ### crea una lista de los archivos en la ruta (no / yes)

    for imagePath in ( list_labels): ### recorre cada carpeta de la ruta ingresada
        
        files_list=listdir(imagePath) ### crea una lista con todos los archivos
        for item in tqdm(files_list): ### le pone contador a la lista: tqdm
            file = join(imagePath, item) ## crea ruta del archivo
            if file[-1] =='g': ### verificar que se imágen extensión jpg o jpeg
                img = cv2.imread(file) ### cargar archivo
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB) ### invierte el orden de los colores en el array para usar el más estándar RGB
                img = cv2.resize(img ,(width,width)) ### cambia resolución de imágnenes
                rawImgs.append(img) ###adiciona imágen al array final
                names.append(file)
                l = imagePath.split('/')[2] ### identificar en qué carpeta está
                if l == 'benign':  ### verificar en qué carpeta está para asignar el label
                    labels.append([0])
                elif l == 'malignant':
                    labels.append([1])
    return np.array(rawImgs), np.array(labels), names


model_metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'AUC': []
}

# Función para calcular y guardar las métricas en el diccionario
def evaluate_model(y_true, y_pred_proba, y_pred_binary, model_name):
    model_metrics['Model'].append(model_name)
    model_metrics['Accuracy'].append(accuracy_score(y_true, y_pred_binary))
    model_metrics['Precision'].append(precision_score(y_true, y_pred_binary))
    model_metrics['Recall'].append(recall_score(y_true, y_pred_binary))
    model_metrics['F1 Score'].append(f1_score(y_true, y_pred_binary))
    model_metrics['AUC'].append(roc_auc_score(y_true, y_pred_proba))