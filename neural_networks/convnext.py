import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ConvNeXtLarge

# CONFIGURATION
IMG_HEIGHT = 150  
IMG_WIDTH = 150  
BATCH_SIZE = 256
NUM_CLASSES = 8   
EPOCHS = 20
DATASET_PATH = '../dataset/'


# PREPROCESSING
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # NORMALIZE
    validation_split=0.2
)


# SPLIT TRAINING AND VALIDATION
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
base_model = ConvNeXtLarge(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Congela las capas iniciales

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    
)

# TRAINING
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data
)

# EVALUATION
loss, accuracy = model.evaluate(validation_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Save model
model.save('almond_disease_classifier_convnext.keras')

# Get true labels and predictions
true_labels = validation_data.classes
class_labels = list(validation_data.class_indices.keys())

# Get predictions
predictions = model.predict(validation_data)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))
