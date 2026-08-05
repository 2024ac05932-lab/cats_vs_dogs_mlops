import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

os.makedirs('reports/figures', exist_ok=True)

# Load model
model = load_model('models/cats_dogs_model.h5')

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory('data/processed/test/',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary',
                                            shuffle=False)

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc:.4f}")

# Predictions and confusion matrix
preds = (model.predict(test_gen) > 0.5).astype(int)
cm = confusion_matrix(test_gen.classes, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('reports/figures/confusion_matrix.png')
print("📊 Confusion matrix saved to reports/figures/confusion_matrix.png")