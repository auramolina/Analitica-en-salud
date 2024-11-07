import numpy as np
import pandas as pd
import tensorflow as tf

import os
from os import listdir ### para hacer lista de archivos en una ruta
from tqdm import tqdm  ### para crear contador en un for para ver evolución
import cv2 ### para leer imagenes jpg
import xgboost as xgb
from sklearn.decomposition import PCA

import sys
sys.path.append('utils/')

### Despliegue XgBoost ####




if __name__ == "__main__":

    # Ruta al directorio Despliegue
    path = 'Despliegue/'  

    # Verificar los archivos en la ruta "Despliegue"
    print(listdir(path))

    x = []  # Lista para almacenar las imágenes
    names = []

    list_labels = [path + f for f in listdir(path)]  # Crear lista con los archivos en la ruta

    # Procesar cada imagen
    for imagePath in tqdm(list_labels):
        if imagePath.lower().endswith(('.jpg', '.jpeg')):  # Solo imágenes .jpg o .jpeg
            img = cv2.imread(imagePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))  # Redimensionar a 100x100
            x.append(img)
            names.append(imagePath)

    x = np.array(x)  # Convertir lista de imágenes a array numpy
    x = x.astype('float32') / 255  # Escalar datos de imagen entre 0 y 1

    # Aplicar PCA (reducción de dimensionalidad)
    pca = PCA(n_components=100)  # Usar 100 componentes principales
    x_pca = pca.fit_transform(x.reshape(len(x), -1))  # Aplanar las imágenes y reducir dimensiones

    # Cargar el modelo de XGBoost (utilizando ruta explícita)
    model_path = 'xgb_model.model'  # Ruta del modelo
    modelo = xgb.Booster()
    modelo.load_model(model_path)

    # Verifica el número de características que espera el modelo
    print(f"Características esperadas por el modelo: {modelo.num_features()}")

    # Realizar predicciones con las imágenes transformadas por PCA
    dmatrix = xgb.DMatrix(x_pca)  # Usar las características reducidas por PCA para la predicción
    prob = modelo.predict(dmatrix)

    # Clasificar según probabilidad
    clase = ['lunar Maligno' if p >= 0.70 else 'lunar Benigno' if p <= 0.20 else "Revisión manual" for p in prob]

    # Mensajes personalizados
    mensaje_no_tumor = '''¡¡Buenas noticias! La evaluación de su lunar indica que es benigno y no presenta signos de malignidad.
    Puede sentirse tranquilo/a, ya que su salud está en buen estado. Sin embargo, le recomendamos continuar con chequeos regulares para mantener su bienestar.'''

    mensaje_revision_manual = '''La evaluación de su lunar en la piel ha arrojado resultados que requieren una revisión manual por parte de un especialista. 
    Esto puede deberse a varios factores y no necesariamente indica malignidad. Le recomendamos que se ponga en contacto con su médico lo antes posible para obtener más información y orientación.'''

    mensaje_si_tumor = '''Lamentamos informarle que hemos detectado ciertas anomalías en la evaluación de su lunar que sugieren la posibilidad de malignidad.
    Sabemos que esta noticia puede ser abrumadora, pero es crucial que actúe con prontitud. Le recomendamos que se comunique de inmediato con su médico para realizar una evaluación detallada y discutir las opciones de tratamiento.
    Estamos aquí para apoyarle en cada paso del proceso. Su bienestar es nuestra prioridad.'''

    # Asignar mensaje según clase
    mensaje = [mensaje_si_tumor if p >= 0.70 else mensaje_no_tumor if p <= 0.20 else mensaje_revision_manual for p in prob]
    
    # Procesar nombres de archivos
    files2 = [name.rsplit('.', 1)[0] for name in names]
    files2 = [name.rsplit('/')[-1] for name in files2]

    # Crear diccionario de resultados
    res_dict = {
        "Paciente": files2,
        "Clase": clase,
        "Mensaje": mensaje,
        "Probabilidad": [str(round(p * 100, 2)) + '%' for p in prob]
    }

    # Generar DataFrame y ordenar resultados
    resultados = pd.DataFrame(res_dict)
    resultados = resultados.sort_values(by=['Clase', 'Probabilidad'], ascending=[True, False])

    # Guardar resultados en un archivo Excel
    resultados.to_excel('salidas/resultados_xgb.xlsx', index=False)