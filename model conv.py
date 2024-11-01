import numpy as np
import joblib  # Para cargar arrays
import tensorflow as tf
from sklearn import metrics  # Para analizar el modelo
import pandas as pd
import keras_tuner as kt  # Asegúrate de haber instalado keras-tuner

# Cargar bases procesadas
x_train = joblib.load('salidas/x_train.pkl')
y_train = joblib.load('salidas/y_train.pkl')
x_test = joblib.load('salidas/x_test.pkl')
y_test = joblib.load('salidas/y_test.pkl')

# Verificar el primer elemento de x_train
print(x_train[0])

# Preprocesamiento
x_train = x_train.astype('float32')  # Cambiar a tipo float32 para escalado
x_test = x_test.astype('float32')

# Escalar datos entre 0 y 1
x_train /= 255
x_test /= 255

# Verificar tamaños
print(x_train.shape, x_test.shape)
print(np.product(x_train[1].shape))  # Cantidad de variables por imagen

# Verificar distribución de clases
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# Definir y compilar el modelo CNN
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
cnn_model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
cnn_model.summary()

# Probar una red con regularización L2
reg_strength = 0.001
dropout_rate = 0.1

cnn_model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo con regularización
cnn_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC', "accuracy"])
cnn_model2.fit(x_train, y_train, batch_size=100, epochs=3, validation_data=(x_test, y_test))

# Evaluación del modelo
pred_test1 = (cnn_model2.predict(x_test) >= 0.98).astype('int')
cm = metrics.confusion_matrix(y_test, pred_test1, labels=[1, 0])
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['tumor', 'No_tumor'])
disp.plot()
print(metrics.classification_report(y_test, pred_test1))

# Afinar hiperparámetros
hp = kt.HyperParameters()

def build_model(hp):
    dropout_rate = hp.Float('DO', min_value=0.05, max_value=0.2, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.0005, step=0.0001)
    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) if optimizer == 'adam' else tf.keras.optimizers.SGD(learning_rate=0.001)
    
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["Recall"])
    return model

# Configuración del tuner
tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=True,
    objective=kt.Objective("recall", direction="max"),
    max_trials=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

# Búsqueda de hiperparámetros
tuner.search(x_train, y_train, epochs=3, validation_data=(x_test, y_test), batch_size=100)

# Seleccionar el mejor modelo
fc_best_model = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()
fc_best_model.summary()

# Evaluación del mejor modelo
test_loss, test_auc = fc_best_model.evaluate(x_test, y_test)
pred_test = (fc_best_model.predict(x_test) >= 0.50).astype('int')

# Exportar el modelo afinado
fc_best_model.save('Analitica-en-salud/best_model.h5')

# Evaluar el modelo afinado
pred_test1 = (fc_best_model.predict(x_test) >= 0.98).astype('int')
cm = metrics.confusion_matrix(y_test, pred_test1, labels=[1, 0])
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['tumor', 'No_tumor'])
disp.plot()
print(metrics.classification_report(y_test, pred_test1))
