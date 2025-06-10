# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Path to the dataset folder (must contain your specific two types subfolders)
data = 'Car-Bike-Dataset'

# Creating a data generator for training data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Normalize pixel values to [0, 1]
    validation_split=0.2,         # Reserve 20% of data for validation
    rotation_range=40,            # Random rotation (±40 degrees)
    width_shift_range=0.2,        # Horizontal shift (±20%)
    height_shift_range=0.2,       # Vertical shift (±20%)
    shear_range=0.2,              # Shearing transformations
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Randomly flip images horizontally
    fill_mode='nearest'           # Fill in new pixels after transform
)

# Data generator for validation (no augmentation, only rescaling)
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create training data generator
train_generator = train_datagen.flow_from_directory(
    data,
    target_size=(150, 150),       # Resize all images to 150x150
    batch_size=32,
    class_mode='binary',          # Binary classification: Car or Bike
    subset='training'
)

# Create validation data generator
validation_generator = validation_datagen.flow_from_directory(
    data,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Building the CNN model
model = models.Sequential()

# First Convolutional Block
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolutional Block
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Block
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fourth Convolutional Block
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))    # Fully connected layer
model.add(layers.Dense(1, activation='sigmoid'))   # Output layer (binary classification)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50
)

# Extracting training history for plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

# Plotting accuracy and loss over epochs
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Function to make prediction on a single image
def predict_image(model, img_path):
    """
    Loads an image, preprocesses it, and predicts whether it is a Car or a Bike.
    """
    img = image.load_img(img_path, target_size=(150, 150))     # Load image
    img_array = image.img_to_array(img)                         # Convert to array
    img_array = np.expand_dims(img_array, axis=0)               # Add batch dimension
    img_array /= 255.0                                          # Normalize

    prediction = model.predict(img_array)                       # Make prediction

    # Display prediction with confidence
    if prediction[0] > 0.5:
        print(f"The image is predicted to be a (item-1) with confidence {prediction[0][0]:.2f}")
    else:
        print(f"The image is predicted to be a (item-2) with confidence {1 - prediction[0][0]:.2f}")

# Test prediction with a sample image
predict_image(model, 'abc')
