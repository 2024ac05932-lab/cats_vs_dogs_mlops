import os
import mlflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# Set MLflow experiment
mlflow.set_experiment("Cats_vs_Dogs_Classifier")

# Parameters
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = (128, 128)

with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("img_size", IMG_SIZE)

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
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=EPOCHS)

    # Log metrics per epoch
    for epoch in range(EPOCHS):
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/cats_dogs_model.keras')
    mlflow.log_artifact('models/cats_dogs_model.keras', artifact_path="models")

    print(f"✅ Training complete. MLflow Run ID: {run.info.run_id}")