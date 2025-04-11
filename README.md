# wood-classification
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

# Define model architecture
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(5, activation='softmax')  # Assuming 5 types of wood
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

# Load trained model or create a new one
model = create_model()

# Example prediction
def predict_wood(image_path):
    classes = ['Oak', 'Pine', 'Maple', 'Birch', 'Walnut']  # Example wood types
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    
    print(f"Predicted Wood Type: {predicted_class}")
    return predicted_class

# Example usage
# predict_wood("example_wood.jpg")
