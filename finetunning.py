import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model


# Configuraciones básicas
IMG_HEIGHT = 150  # Altura a la que redimensionaremos
IMG_WIDTH = 150   # Anchura a la que redimensionaremos
BATCH_SIZE = 64
NUM_CLASSES = 7   # Número de categorías de enfermedades
EPOCHS = 20

# Ruta a las imágenes (organizadas en carpetas según las categorías)
DATASET_PATH = 'dataset/'

# Preprocesamiento (solo normalización)
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



# Cargar MobileNetV2 preentrenada
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Congelar las capas del modelo base
base_model.trainable = False

# Añadir capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Modelo final
model = Model(inputs=base_model.input, outputs=output)

# Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Resumen del modelo
model.summary()

# Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Métrica a monitorear
    patience=5,          # Número de épocas sin mejora antes de detener
    restore_best_weights=True  # Restaurar los mejores pesos al final
)


# Entrenamiento del modelo
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data
)

# Evaluación del modelo
loss, accuracy = model.evaluate(validation_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Guardar el modelo
model.save('almendro_disease_classifier.keras')

# Gráfica de precisión
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Gráfica de pérdida
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


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