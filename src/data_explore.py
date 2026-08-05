import os
import cv2
import matplotlib.pyplot as plt

# Count files
cat_path = 'data/raw/cat/'
dog_path = 'data/raw/dog/'
print(f"Cats: {len(os.listdir(cat_path))}, Dogs: {len(os.listdir(dog_path))}")

# Check for corrupted images
for folder in [cat_path, dog_path]:
    for img_name in os.listdir(folder)[:100]:  # Check first 100
        try:
            img = cv2.imread(os.path.join(folder, img_name))
            if img is None:
                print(f"Corrupted: {img_name}")
        except:
            print(f"Error reading: {img_name}")

# Visualize a sample
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(cv2.imread(os.path.join(cat_path, os.listdir(cat_path)[0]))[:, :, ::-1])
axes[0].set_title('Cat')
axes[1].imshow(cv2.imread(os.path.join(dog_path, os.listdir(dog_path)[0]))[:, :, ::-1])
axes[1].set_title('Dog')
plt.savefig('reports/figures/sample_images.png')  # Save report
print("EDA complete. Check reports/figures/")