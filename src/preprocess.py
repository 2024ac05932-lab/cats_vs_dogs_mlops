import os
import shutil
from sklearn.model_selection import train_test_split
import random

# Define paths
raw_cat = 'data/raw/cat/'
raw_dog = 'data/raw/dog/'
processed_base = 'data/processed/'

# Create directories
for split in ['train', 'val', 'test']:
    for cls in ['cat', 'dog']:
        os.makedirs(os.path.join(processed_base, split, cls), exist_ok=True)

# Get all images and shuffle
cat_imgs = [(f, 'cat') for f in os.listdir(raw_cat)]
dog_imgs = [(f, 'dog') for f in os.listdir(raw_dog)]
all_data = cat_imgs + dog_imgs
random.shuffle(all_data)

# Split: 70% train, 15% val, 15% test
train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Copy function
def copy_files(data_list, split_name):
    for filename, label in data_list:
        src = raw_cat if label == 'cat' else raw_dog
        dst = os.path.join(processed_base, split_name, label, filename)
        shutil.copy(os.path.join(src, filename), dst)

copy_files(train_data, 'train')
copy_files(val_data, 'val')
copy_files(test_data, 'test')

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")