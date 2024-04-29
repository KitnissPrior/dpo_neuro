import pandas as pd
import numpy as np
import os
#убирает предупреждения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input

# Load the dataset
data_path = 'D:/ИРИТ/3 курс/нейронки/messidor_data.csv'
images_folder_path = 'D:/ИРИТ/3 курс/нейронки/messidor-2/messidor-2/preprocess/'

# Assuming the messidor-data.csv is structured with columns 'id_code', 'diagnosis', 'adjudicated_time', 'adjudicated_duration'
data = pd.read_csv(data_path)

# Displaying the first few rows of the dataset
print(data.head())

# Assuming that the images are stored in the folder 'images/' and are named as per 'id_code' in the data
data['path'] = images_folder_path + data['id_code'].astype(str) + '.jpeg'

# Splitting the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['diagnosis'], random_state=42)

# Image preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Training generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=None,
    x_col='path',
    y_col='diagnosis',
    class_mode='raw',
    target_size=(224, 224),
    batch_size=32
)

# Testing generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    directory=None,
    x_col='path',
    y_col='diagnosis',
    class_mode='raw',
    target_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Определение модели с использованием Input в качестве первого слоя
model = tf.keras.models.Sequential([
    Input(shape=(224, 224, 3)), # Добавление Input слоя
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax') # Output layer for 5 classes
])

# Compile the model
# Исправление
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Training the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    verbose=2
)

# Plotting training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()

# Evaluating the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Prediction and Classification Report
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(confusion_matrix(true_classes, predicted_classes))
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

