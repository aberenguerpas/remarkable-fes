import numpy as np
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# CONFIGURATION
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 256
NUM_CLASSES = 8
EPOCHS = 20
DATASET_PATH = '../dataset/'

# PREPROCESSING
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalization
    validation_split=0.2
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

# LOAD MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# COMPILATION
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=5, 
    restore_best_weights=True
)

# TRAINING
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=validation_data,
    callbacks=[early_stopping]
)

# EVALUATION
loss, accuracy = model.evaluate(validation_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# SAVE MODEL
model.save('almendro_disease_classifier_mobilenet.keras')

# Get true labels and predictions
true_labels = validation_data.classes
class_labels = list(validation_data.class_indices.keys())

# Get predictions
predictions = model.predict(validation_data)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))