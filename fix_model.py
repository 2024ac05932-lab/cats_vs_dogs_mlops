import os
# Force use of legacy Keras (Keras 2) for saving
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model (it will use legacy Keras if the environment variable is set)
model = load_model('models/cats_dogs_model.keras')   # or '.h5' if that works

# Save in HDF5 format using the legacy backend
model.save('models/cats_dogs_model_legacy.h5', save_format='h5')
print("✅ Model re‑saved as cats_dogs_model_legacy.h5")