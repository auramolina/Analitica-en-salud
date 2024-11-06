import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# Define paths
train_dir = 'Data/train/'
test_dir = 'Data/test/'

# Load ResNet50 model pre-trained on ImageNet, excluding the top layer
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Set up image data generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create a generator for the training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Extract features using ResNet50
features = base_model.predict(train_generator)
labels = train_generator.classes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print('------------------XGBOOST---------------------------')
test_auc = roc_auc_score(y_test, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
print(f'AUC - Test XGBoost: {test_auc:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualización de la matriz de confusión
# cm = confusion_matrix(test_labels, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
# disp.plot(cmap="Blues")
# plt.title("Matriz de Confusión - XGBoost")
# plt.xlabel("Predicción")
# plt.ylabel("Verdad Real")
# plt.show()
# matriz de confusión test
from sklearn import metrics
pred_test = (xgb_model.predict(X_test) > 0.70).astype('int')
cm = metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp = metrics.ConfusionMatrixDisplay(cm,display_labels=['Malignant', 'Benign'])
disp.plot()
# print(metrics.classification_report(y_test, pred_test))

# Exportar modelo
xgb_model.save_model("xgb_model.model")
