import idx2numpy
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

print("Loading EMNIST Letters Dataset...")


# LOAD DATA

x_train = idx2numpy.convert_from_file(
    "data/emnist-letters-train-images-idx3-ubyte"
)

y_train = idx2numpy.convert_from_file(
    "data/emnist-letters-train-labels-idx1-ubyte"
)

x_test = idx2numpy.convert_from_file(
    "data/emnist-letters-test-images-idx3-ubyte"
)

y_test = idx2numpy.convert_from_file(
    "data/emnist-letters-test-labels-idx1-ubyte"
)

# FIX LABELS

y_train = y_train - 1
y_test = y_test - 1

# FIX EMNIST ORIENTATION

print("Fixing EMNIST Orientation...")

x_train = np.array([
    np.fliplr(np.rot90(img))
    for img in x_train
])

x_test = np.array([
    np.fliplr(np.rot90(img))
    for img in x_test
])

# NORMALIZE

x_train = x_train / 255.0
x_test = x_test / 255.0

# RESHAPE

x_train = x_train.reshape(
    -1,
    28,
    28,
    1
)

x_test = x_test.reshape(
    -1,
    28,
    28,
    1
)

print("Training Shape:", x_train.shape)
print("Labels Shape:", y_train.shape)

# BUILD CNN

print("Building CNN Model...")

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(28,28,1)
    ),

    BatchNormalization(),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    BatchNormalization(),

    MaxPooling2D((2,2)),

    Dropout(0.25),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        26,
        activation='softmax'
    )
])

# COMPILE MODEL

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully!")

# TRAIN MODEL

print("Starting Training...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

# EVALUATE MODEL

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print(f"\nTest Accuracy: {accuracy:.4f}")

# SAVE MODEL

model.save(
    "models/letter_model.h5"
)

print("Model Saved Successfully!")