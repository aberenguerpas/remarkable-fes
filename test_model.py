import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model





# Configuraciones básicas
IMG_HEIGHT = 150  # Altura a la que redimensionaremos
IMG_WIDTH = 150   # Anchura a la que redimensionaremos
NUM_CLASSES = 7
BATCH_SIZE = 128
DATASET_PATH = 'dataset/'

datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalización de imágenes
    validation_split=0.2  # División entre entrenamiento y validación
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Redimensionado al cargar
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Redimensionado al cargar
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


#Cargar modelo .keras
model = load_model('./almendro_disease_classifier.keras')


# Evaluación del modelo
loss, accuracy = model.evaluate(validation_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Get true labels and predictions
true_labels = validation_data.classes
class_labels = list(validation_data.class_indices.keys())

# Get predictions
predictions = model.predict(validation_data)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))
