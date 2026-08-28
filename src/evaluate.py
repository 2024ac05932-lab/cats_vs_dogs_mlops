import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import mlflow

mlflow.set_experiment("Cats_vs_Dogs_Classifier")

with mlflow.start_run() as run:
    # Load model
    model = load_model('models/cats_dogs_model.keras')
    mlflow.log_artifact('models/cats_dogs_model.keras', artifact_path="models")

    # Test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory('data/processed/test/',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary',
                                                shuffle=False)

    # Evaluate
    loss, acc = model.evaluate(test_gen)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_loss", loss)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Confusion matrix
    preds = (model.predict(test_gen) > 0.5).astype(int)
    cm = confusion_matrix(test_gen.classes, preds)

    # Save and log confusion matrix figure
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('reports/figures/confusion_matrix.png')
    mlflow.log_artifact('reports/figures/confusion_matrix.png', artifact_path="figures")
    print("📊 Confusion matrix saved and logged to MLflow")