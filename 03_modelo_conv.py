import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

####instalar paquete !pip install keras-tuner
import keras_tuner as kt


### cargar bases_procesadas ####
x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

#x_train = joblib.load('C:\Users\Gilber\Desktop\Analitica-en-salud\Analitica-en-salud\Salidas')
#y_train = joblib.load('C:\Users\Gilber\Desktop\Analitica-en-salud\Analitica-en-salud\Salidas')
#x_test = joblib.load('C:\Users\Gilber\Desktop\Analitica-en-salud\Analitica-en-salud\Salidas')
#y_test = joblib.load('C:\Users\Gilber\Desktop\Analitica-en-salud\Analitica-en-salud\Salidas')

x_train[0]

############################################################
################ Preprocesamiento ##############
############################################################

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train.max()
x_train.min()


x_train /=255 ### escalarlo para que quede entre 0 y 1, con base en el valor máximo
x_test /=255

###### verificar tamaños

x_train.shape
x_test.shape

np.product(x_train[1].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

##########################################################
################ Redes convolucionales ###################
##########################################################

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

# Train the model for 30 epochs
cnn_model.fit(x_train, y_train, batch_size=100, epochs=30, validation_data=(x_test, y_test))


cnn_model.summary()

#######probar una red con regulzarización L2 Para ver si hay alguna diferencia debido a que los datos dieron muy bien
#######probar una red con regulzarización L2
reg_strength = 0.001

###########Estrategias a usar: regilarization usar una a la vez para ver impacto
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

# Compile the model with binary cross-entropy loss and Adam optimizer
cnn_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC',"accuracy"])

# Train the model for 10 epochs
#cnn_model2.fit(x_train, y_train, batch_size=100, epochs=30, validation_data=(x_test, y_test))

#pred_test1=(cnn_model2.predict(x_test) >= 0.98).astype('int')
#cm=metrics.confusion_matrix(y_test,pred_test1, labels=[1,0])
#disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['tumor', 'No_tumor'])
#disp.plot()

#print(metrics.classification_report(y_test, pred_test1))

#####################################################
###### afinar hiperparameter ########################
#####################################################


##para ver si disminuye los falsos positivos se decide afinar los hiper parametros
##### función con definicion de hiperparámetros a afinar
hp = kt.HyperParameters()

def build_model(hp):
    
    dropout_rate=hp.Float('DO', min_value=0.05, max_value= 0.5, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0001, max_value=0.0009, step=0.0001)
    optimizer = hp.Choice('optimizer', ['adam', 'sgd']) ### en el contexto no se debería afinar
    
    model= tf.keras.Sequential([
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
    
    
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
   
    model.compile(
        optimizer=opt, loss="binary_crossentropy", metrics=["Recall", "AUC"],
    )
    
    
    return model



###########

tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=True, 
    objective=kt.Objective("Recall", direction="max"),
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld", 
)



tuner.search(x_train, y_train, epochs=30, validation_data=(x_test, y_test), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]


tuner.results_summary()
fc_best_model.summary()


test_loss, test_auc=fc_best_model.evaluate(x_test, y_test)
pred_test=(fc_best_model.predict(x_test)>=0.5).astype('int')
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()

print(metrics.classification_report(y_test, pred_test1))

#################### exportar modelo afinado ##############
fc_best_model.save('best_model.h5')
#################### exportar modelo afinado ##############