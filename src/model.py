import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Import matplotlib

def create_improved_model(l2_reg=0.001):
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg), input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 4
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Fully connected layer 1
    model.add(Dense(512, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(7, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the dataset
# Note: You might need to adjust the path to your file
try:
    data = pd.read_csv('/content/drive/MyDrive/fer2013.csv')
except FileNotFoundError:
    print("Error: 'fer2013.csv' not found. Please update the file path.")
    # As a placeholder, create dummy data to allow the script to run
    data = pd.DataFrame({
        'emotion': np.random.randint(0, 7, 100),
        'pixels': [' '.join(map(str, np.random.randint(0, 256, 48*48))) for _ in range(100)]
    })


# Preprocess the data
X = []
y = []
for index, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='float32')
    if pixels.shape[0] == 48*48:
        X.append(pixels.reshape(48, 48, 1))
        y.append(row['emotion'])
    else:
        print(f"Skipping row {index} due to incorrect number of pixels: {pixels.shape[0]}")

X = np.array(X) / 255.0
y = to_categorical(np.array(y), num_classes=7)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# --- Create and Train Model ---
model = create_improved_model()
model.summary()

batch_size = 32
epochs = 100

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the final model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Final Test Loss: {loss:.4f}')
print(f'Final Test Accuracy: {accuracy:.4f}')

# Save the improved model
model.save('emotion_detection_model_improved.h5')

# --- PLOTTING LOSS AND ACCURACY ---
# The 'history' object holds the training history.
# We can access the metrics from history.history

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training & validation accuracy values
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')
ax1.grid(True)

# Plot training & validation loss values
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')
ax2.grid(True)

# Show the plots
plt.tight_layout()
plt.show()