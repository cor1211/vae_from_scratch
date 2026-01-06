from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as F 
from PIL import Image
from pathlib import Path
import yaml
import sys
from argparse import ArgumentParser
import random

def load_config(config_path: str):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    
    except FileNotFoundError:
        print('The path of config is not valid')
        sys.exit(1)


class CartoonFacesDataset(Dataset):
    # Each subfolders (lablels) have 10k images -> Train/test: 80/20
    def __init__(self, train: bool = True, transform = None, cfg_data: dict = None):
        self.transform = transform
        self.path_list = []
        self.train_percent = cfg_data['train_percent']
        root = cfg_data['root']
        self.train = train

        for folder in sorted(os.listdir(root)):
            # print(f'Loading folder: {folder}')
            folder_path = os.path.join(root, folder)
            image_quantity = len(os.listdir(folder_path))
            end_train = image_quantity * self.train_percent
            end_valid = image_quantity - end_train

            if self.train: #  get path from idx: 0 to 80% each folder
                for idx, image_name in enumerate(sorted(os.listdir(folder_path))):
                    image_path = os.path.join(folder_path, image_name)
                    self.path_list.append(image_path)
                    idx+=1
                    if idx >= end_train:
                        break

            else: # valid
                for idx, image_name in enumerate(sorted(os.listdir(folder_path), reverse=True)):
                    image_path = os.path.join(folder_path, image_name)
                    self.path_list.append(image_path)
                    if idx >= end_valid - 1:
                        break

    def __len__(self):
        return len(self.path_list)
    

    def __getitem__(self, idx):
        # Open img
        image = Image.open(self.path_list[idx]).convert('RGB')
        # Down-sample 500 -> 256
        image = image.resize(size=(128, 128), resample=Image.Resampling.BICUBIC)
        # Apply augmentations only for the training set
        if self.train:
            if random.random() > 0.5:
                image = F.hflip(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image
    

if __name__ == '__main__':
    parser = ArgumentParser(prog='Dataset arguments')
    parser.add_argument('--config_path', type=str, required=True, help='Config_path to base_config.yaml')
    args = parser.parse_args()

    #-------Load yaml file-----------
    config_dict = load_config(args.config_path)
    cfg_data = config_dict['data']

    
    #-----Initialize------
    train_set = CartoonFacesDataset(cfg_data=cfg_data, train=True)
    valid_set = CartoonFacesDataset(cfg_data=cfg_data, train=False)


    #--------View---------
    print(len(train_set))
    print(len(valid_set))

    train_set[0].show()
    valid_set[0].show()