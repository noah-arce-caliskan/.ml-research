import os
import shutil
import random

def split_celeb_data(source_folder, output_folder, test_split=0.5):
    train_dir = os.path.join(output_folder, "train")
    test_dir = os.path.join(output_folder, "test")

    celebrities = [folder for folder in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, folder))]
    
    random.shuffle(celebrities)
    split_index = int(len(celebrities) * test_split)
    test_celebrities = celebrities[:split_index]
    train_celebrities = celebrities[split_index:]

    for celeb in train_celebrities:
        shutil.copytree(os.path.join(source_folder, celeb), os.path.join(train_dir, celeb))
    for celeb in test_celebrities:
        shutil.copytree(os.path.join(source_folder, celeb), os.path.join(test_dir, celeb))

    print(f"Data split completed. Train: {len(train_celebrities)} celebrities, Test: {len(test_celebrities)} celebrities.")
    print(f"Train folder: {train_dir}")
    print(f"Test folder: {test_dir}")


source_folder = "/workspace/noah/lfw/lfw-deepfunneled"
output_folder = "/workspace/noah/lfw/customsplit"
split_celeb_data(source_folder, output_folder)
