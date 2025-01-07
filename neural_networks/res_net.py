import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# CONFIGURATION
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 20
DATASET_PATH = '../dataset/'

# NETWORK DEFINITION
def residual_block(inputs, filters, stride=1, use_projection=False):
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
    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    filters = 64
    for i, num_blocks in enumerate(num_blocks_per_layer):
        for j in range(num_blocks):
            stride = 2 if j == 0 and i > 0 else 1
            use_projection = True if j == 0 and (i > 0 or filters != 64) else False
            x = residual_block(x, filters, stride=stride, use_projection=use_projection)
        filters *= 2

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)  

resnet = build_resnet(input_shape, NUM_CLASSES)

resnet.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# PREPROCESSING
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
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
    subset='validation',
    shuffle=False
)

# TRAINING
history = resnet.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data
)

# SAVE MODEL
resnet.save('almendro_disease_classifier_resnet.keras')


# EVALUATION
loss, accuracy = resnet.evaluate(validation_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Get true labels and predictions
true_labels = validation_data.classes
class_labels = list(validation_data.class_indices.keys())

# Get predictions
predictions = resnet.predict(validation_data)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))