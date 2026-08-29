import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import mlflow

# Set MLflow experiment name
mlflow.set_experiment("Cats_vs_Dogs_Classifier")

def main():
    with mlflow.start_run() as run:
        # Load model and log as artifact
        model_path = 'models/cats_dogs_model.keras'
        model = load_model(model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        # Test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            'data/processed/test/',
            target_size=(128, 128),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )

        # Evaluate model performance
        loss, acc = model.evaluate(test_gen)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_loss", loss)
        print(f"✅ Test Accuracy: {acc:.4f}")

        # Generate confusion matrix
        preds = (model.predict(test_gen) > 0.5).astype(int)
        cm = confusion_matrix(test_gen.classes, preds)

        # Save and log confusion matrix figure
        os.makedirs('reports/figures', exist_ok=True)
        fig_path = 'reports/figures/confusion_matrix.png'
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(fig_path)
        plt.close()

        mlflow.log_artifact(fig_path, artifact_path="figures")
        print("📊 Confusion matrix saved and logged to MLflow")

if __name__ == "__main__":
    main()