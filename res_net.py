from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Configuraciones básicas
IMG_HEIGHT = 150  # Altura a la que redimensionaremos
IMG_WIDTH = 150   # Anchura a la que redimensionaremos
BATCH_SIZE = 32
NUM_CLASSES = 7   # Número de categorías de enfermedades
EPOCHS = 20

def residual_block(inputs, filters, stride=1, use_projection=False):
    """Implementa un bloque residual básico."""
    shortcut = inputs

    # Primera capa convolucional
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Segunda capa convolucional
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Proyección para ajustar dimensiones si es necesario
    if use_projection or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(inputs)
        shortcut = BatchNormalization()(shortcut)

    # Suma de la entrada y la salida
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def build_resnet(input_shape, num_classes, num_blocks_per_layer=[2, 2, 2, 2]):
    """Construye una ResNet personalizada."""
    inputs = Input(shape=input_shape)

    # Capa inicial (convolución y max pooling)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # Bloques residuales
    filters = 64
    for i, num_blocks in enumerate(num_blocks_per_layer):
        for j in range(num_blocks):
            stride = 2 if j == 0 and i > 0 else 1  # Reduce la resolución al inicio de cada grupo
            use_projection = True if j == 0 and (i > 0 or filters != 64) else False
            x = residual_block(x, filters, stride=stride, use_projection=use_projection)
        filters *= 2

    # Pooling y capa de salida
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Parámetros del modelo
input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)  # Cambia esto según el tamaño de tus imágenes
num_classes = 7  # Cambia según tu problema
resnet = build_resnet(input_shape, num_classes)

# Resumen del modelo
resnet.summary()

resnet.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

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

# Entrenamiento del modelo
history = resnet.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data
)

# Evaluación del modelo
loss, accuracy = resnet.evaluate(validation_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Guardar el modelo
resnet.save('almendro_disease_classifier_resnet.keras')

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
