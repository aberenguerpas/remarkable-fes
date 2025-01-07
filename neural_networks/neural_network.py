import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CONFIGURATION
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 128
NUM_CLASSES = 8
EPOCHS = 20
DATASET_PATH = '../dataset/'


# PREPROCESSING
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
)
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


# MODEL
model = Sequential([
   
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),


    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),  


    Conv2D(128, (5, 5), activation='relu'),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),  
    Dropout(0.4),

   
    Conv2D(256, (5, 5), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),

    GlobalAveragePooling2D(),


    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(), 
    Dropout(0.1),


    Dense(NUM_CLASSES, activation='softmax')  # Para 8 clases
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# TRAINING
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# EVALUATION
loss, accuracy = model.evaluate(validation_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# SAVE MODEL
model.save('almendro_disease_classifier_NN.keras')

# Get true labels and predictions
true_labels = validation_data.classes
class_labels = list(validation_data.class_indices.keys())

# Get predictions
predictions = model.predict(validation_data)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))
