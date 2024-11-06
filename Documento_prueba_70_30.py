from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import tensorflow as tf
from sklearn import metrics # para analizar modelo
# Cargar datos de entrenamiento y prueba originales
x_train = joblib.load('C:\\Users\\Gilber\\Desktop\\Analitica-en-salud\\Analitica-en-salud\\Salidas\\x_train.pkl')
y_train = joblib.load('C:\\Users\\Gilber\\Desktop\\Analitica-en-salud\\Analitica-en-salud\\Salidas\\y_train.pkl')
x_test = joblib.load('C:\\Users\\Gilber\\Desktop\\Analitica-en-salud\\Analitica-en-salud\\Salidas\\x_test.pkl')
y_test = joblib.load('C:\\Users\\Gilber\\Desktop\\Analitica-en-salud\\Analitica-en-salud\\Salidas\\y_test.pkl')

# Combinar los datos nuevamente
x_combined = np.concatenate((x_train, x_test), axis=0)
y_combined = np.concatenate((y_train, y_test), axis=0)

# Dividir los datos en 70% para entrenamiento y 30% para prueba
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined)

# Verificar los tamaños de los conjuntos resultantes
print("Tamaño del nuevo conjunto de entrenamiento:", x_train_new.shape, y_train_new.shape)
print("Tamaño del nuevo conjunto de prueba:", x_test_new.shape, y_test_new.shape)

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

# Train the model for 10 epochs
cnn_model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))


cnn_model.summary()

pred_test1=(cnn_model.predict(x_test) >= 0.98).astype('int')
cm=metrics.confusion_matrix(y_test,pred_test1, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['tumor', 'No_tumor'])
disp.plot()