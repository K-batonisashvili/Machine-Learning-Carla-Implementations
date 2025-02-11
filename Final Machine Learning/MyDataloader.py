import glob
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import json
import logging

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyDataLoader(Dataset):
    def __init__(self, data_dir):
        self.img_list = []
        self.depth_list = []
        self.data_list = []
        self.data_dir = data_dir

        # directories
        for run_dir in glob.glob(os.path.join(self.data_dir, 'rc_data', 'run_*')):
            # rgb
            rgb_dir = os.path.join(run_dir, 'rgb')
            self.img_list += glob.glob(os.path.join(rgb_dir, '*.jpg'))

            # depth
            depth_dir = os.path.join(run_dir, 'disparity')
            self.depth_list += glob.glob(os.path.join(depth_dir, '*.png'))

            # json for action
            json_dir = os.path.join(run_dir, 'json')
            self.data_list += glob.glob(os.path.join(json_dir, '*.json'))

        min_length = min(len(self.img_list), len(self.depth_list), len(self.data_list))
        self.img_list = self.img_list[:min_length]
        self.depth_list = self.depth_list[:min_length]
        self.data_list = self.data_list[:min_length]

        # printing out loaded data
        logging.info(f'Loaded {len(self.img_list)} images, {len(self.depth_list)} depth images, and {len(self.data_list)} action files.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            with open(self.data_list[idx], 'r') as f:
                data = json.load(f)

            # extracting throttle for ground truth & predictions
            throttle = torch.tensor([data["Throttle"]], dtype=torch.float32)

            img_path = self.img_list[idx]
            img = read_image(img_path)

            # resizing rgb to 300 by 300 for computational saving
            img = img[:3, :, :] 
            img = transforms.Resize((300, 300))(img) 
            
            normalized_image = img.float() / 255.0  # Normalize image to [0, 1]

            # corresponding depth image for rgb counterpart
            depth_path = self.depth_list[idx]
            depth_img = read_image(depth_path)
            depth_img = depth_img.float() / 255.0  # Normalize depth to [0, 1]

            # also resize to 300 to save computation
            depth_img = transforms.Resize((300, 300))(depth_img)
            combined_image = torch.cat((normalized_image, depth_img), dim=0)

            return combined_image, throttle
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            raise