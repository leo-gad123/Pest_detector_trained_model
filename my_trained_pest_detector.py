import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import requests
import json

# ---------------------------
# 🔹 1. Set dataset directories
# ---------------------------
# Directory structure example:
# dataset/
#    train/
#       ants/
#       beetles/
#       caterpillars/
#    validation/
#       ants/
#       beetles/
#       caterpillars/

train_dir = 'dataset/train'
val_dir = 'dataset/validation'

# ---------------------------
# 🔹 2. Image parameters
# ---------------------------
img_height, img_width = 150, 150
batch_size = 32

# ---------------------------
# 🔹 3. Data generators
# ---------------------------
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# ---------------------------
# 🔹 4. Build CNN model
# ---------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  # Output layer
])

# ---------------------------
# 🔹 5. Compile model
# ---------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# 🔹 6. Firebase connection setup
# ---------------------------
FIREBASE_URL = "https://test-3e174-default-rtdb.firebaseio.com/training_data.json"

def send_to_firebase(epoch, logs):
    """Send training metrics to Firebase Realtime Database."""
    data = {
        "epoch": epoch + 1,
        "accuracy": float(logs.get('accuracy', 0)),
        "val_accuracy": float(logs.get('val_accuracy', 0)),
        "loss": float(logs.get('loss', 0)),
        "val_loss": float(logs.get('val_loss', 0))
    }

    try:
        response = requests.post(FIREBASE_URL, json=data)
        if response.status_code == 200:
            print(f"✅ Data sent to Firebase (Epoch {epoch + 1})")
        else:
            print(f"⚠️ Failed to send data (HTTP {response.status_code})")
    except Exception as e:
        print(f"❌ Firebase error: {e}")

# ---------------------------
# 🔹 7. Create custom callback
# ---------------------------
class FirebaseCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        send_to_firebase(epoch, logs)

# ---------------------------
# 🔹 8. Train the model
# ---------------------------
firebase_callback = FirebaseCallback()

history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[firebase_callback]
)

# ---------------------------
# 🔹 9. Save model (optional)
# ---------------------------
model.save("insect_classifier.h5")

print("✅ Training complete and model saved!")
