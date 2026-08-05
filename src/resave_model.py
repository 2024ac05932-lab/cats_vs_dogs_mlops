import os
from tensorflow.keras.models import load_model

# Load the existing model
model = load_model('models/cats_dogs_model.h5')

# Save it in the new Keras format (recommended)
model.save('models/cats_dogs_model.keras')

print("✅ Model re-saved as 'models/cats_dogs_model.keras'")
print("Now update your API code to load the .keras file instead.")