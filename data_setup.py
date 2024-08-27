import os
import shutil
import torch
import zipfile
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def unzip():
    zip_files = ['test1', 'train']
    ## Run to unzip the file
    for zip_file in zip_files:
        with zipfile.ZipFile("./dogs-vs-cats/{}.zip".format(zip_file),"r") as z:
            z.extractall("./dogs-vs-cats")
            print("{} unzipped".format(zip_file))

    image_path = Path("dogs-vs-cats/")
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    validation_dir = image_path / "validation"
    return image_path, train_dir, test_dir, validation_dir

class OrganizeData():
    def __init__(self, train_dir):
        self.train_dir = train_dir

    def split_images_into_classes(self):
        # Create 'cat' and 'dog' directories inside the 'train' directory
        cat_dir = os.path.join(self.train_dir, 'cat')
        dog_dir = os.path.join(self.train_dir, 'dog')
        os.makedirs(cat_dir, exist_ok=True)
        os.makedirs(dog_dir, exist_ok=True)
        image_path = Path("dogs-vs-cats/")
        train_dir = image_path / "train"
        test_dir = image_path / "test"
        validation_dir = image_path / "validation"
        # Iterate through the files in the 'train' directory
        for filename in os.listdir(train_dir):
            file_path = os.path.join(train_dir, filename)
            
            # Skip if it's not a file (e.g., the newly created directories)
            if os.path.isfile(file_path):
                if filename.startswith('cat'):
                    shutil.move(file_path, os.path.join(cat_dir, filename))
                elif filename.startswith('dog'):
                    shutil.move(file_path, os.path.join(dog_dir, filename))

        print("Images have been successfully split into 'cat' and 'dog' directories.")


    def define_directory_paths(self):
        # Define your directory paths
        base_dir = "dogs-vs-cats"
        train_dir = os.path.join(base_dir, "train")
        test_dir = os.path.join(base_dir, "test")
        validation_dir = os.path.join(base_dir, "validation")
        return train_dir, test_dir, validation_dir
    
    def organize_test_and_validation_images(self):
        train_dir, test_dir, validation_dir = self.define_directory_paths()

        # Create class directories for the test data
        for class_name in ['cat', 'dog']:
            class_test_dir = os.path.join(test_dir, class_name)
            if not os.path.exists(class_test_dir):
                os.makedirs(class_test_dir)

            class_valid_dir = os.path.join(validation_dir, class_name)
            if not os.path.exists(class_valid_dir):
                os.makedirs(class_valid_dir)


        # Organize test and valid images by moving them to the test and valid directory
        for class_name in ['cat', 'dog']:
            
            class_train_dir = os.path.join(train_dir, class_name)
            for filename in os.listdir(class_train_dir):
                if torch.rand(1).item() < 0.15:  # 15% probability
                    source = os.path.join(class_train_dir, filename)
                    destination = os.path.join(test_dir, class_name, filename)
                    shutil.move(source, destination)

            class_train_dir = os.path.join(train_dir, class_name)
            for filename in os.listdir(class_train_dir):

                if torch.rand(1).item() < 0.15:  # 15% probability
                    source = os.path.join(class_train_dir, filename)
                    destination = os.path.join(validation_dir, class_name, filename)
                    shutil.move(source, destination)


img_path, train_dir, test_dir, validation_dir = unzip()
    
organizer = OrganizeData(train_dir="dogs-vs-cats/train")

# arrange cat and dog files directories
organizer.split_images_into_classes()

# arrange test and validation sets
organizer.organize_test_and_validation_images()