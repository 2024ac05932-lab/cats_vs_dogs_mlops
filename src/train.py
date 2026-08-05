import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# Parameters
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = (128, 128)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.15,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory('data/processed/train/',
                                              target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode='binary')
val_gen = val_datagen.flow_from_directory('data/processed/val/',
                                          target_size=IMG_SIZE,
                                          batch_size=BATCH_SIZE,
                                          class_mode='binary')

# Build and train
model = build_model()
print("Training started...")
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=EPOCHS)

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/cats_dogs_model.h5')
print(f"✅ Model saved to models/cats_dogs_model.h5")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")