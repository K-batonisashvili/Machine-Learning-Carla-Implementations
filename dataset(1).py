import glob

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os

import pandas as pd 

from PIL import Image

class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        # Loading images into a list
        self.image_dir = data_dir
        self.image_files = sorted(glob.glob(self.image_dir + '/*.jpg'))  # if these are in jpg we need to change them

        # loading csv file into python object
        csv_file = os.path.join(data_dir, 'drive_data.csv')
        self.drive_data = pd.read_csv(csv_file)

        # tensor flow stuff
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensor
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """

        # Load the image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # Apply transformations

        # Actions taken from the headings of drive data csv
        action_row = self.drive_data.iloc[idx]
        action = torch.tensor([
            action_row['Throttle'],
            action_row['Brake'],
            action_row['Steering'],
        ], dtype=torch.float32)

        return image, action


def get_dataloader(train_folder, batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir=train_folder),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    