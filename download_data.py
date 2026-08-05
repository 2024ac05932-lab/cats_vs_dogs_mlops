import kagglehub
import os
import shutil
import zipfile
import glob

print("Downloading Dogs vs Cats competition files...")
path = kagglehub.competition_download('dogs-vs-cats')
print(f"Downloaded to: {path}")

os.makedirs("data/raw", exist_ok=True)

for file in ["train.zip", "test1.zip", "sampleSubmission.csv"]:
    src = os.path.join(path, file)
    dst = os.path.join("data/raw", file)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {file} to data/raw/")

print("\nExtracting train.zip...")
train_zip = "data/raw/train.zip"
if os.path.exists(train_zip):
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall("data/raw/")
    print("Extracted train.zip")

    os.makedirs("data/raw/cat", exist_ok=True)
    os.makedirs("data/raw/dog", exist_ok=True)

    for file in glob.glob("data/raw/train/cat.*.jpg"):
        shutil.move(file, "data/raw/cat/")
    for file in glob.glob("data/raw/train/dog.*.jpg"):
        shutil.move(file, "data/raw/dog/")

    os.rmdir("data/raw/train")
    os.remove(train_zip)
    print("Organized images into data/raw/cat/ and data/raw/dog/")
else:
    print("train.zip not found - something went wrong.")

print("\nDone! Dataset is ready.")
