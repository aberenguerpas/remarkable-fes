import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

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



# Definición del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),

    Dense(7, activation='softmax')  # 7 clases
])

# Compilación del modelo
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
