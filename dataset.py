from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor, Compose, Normalize
from PIL import Image

class CartoonFacesDataset(Dataset):
    # Each subfolders (lablels) have 10k images -> Train/test: 80/20
    def __init__(self, root:str, train: bool = True, transform = None):
        self.transform = transform
        self.path_list = []
        self.train_percent = 0.8
        
        # root = os.path.join(root, 'train' if train else 'valid') 

        for folder in sorted(os.listdir(root)):
            # print(f'Loading folder: {folder}')
            folder_path = os.path.join(root, folder)
            image_quantity = len(os.listdir(folder_path))
            end_train = image_quantity * self.train_percent
            end_valid = image_quantity - end_train

            if train: #  get path from idx: 0 to 80% each folder
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
        # open img
        print(self.path_list[idx])
        image = Image.open(self.path_list[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image
    
if __name__ == '__main__':
    root = '/mnt/data1tb/vinh/vae_from_scratch/dataset/cartoonset100k_jpg'
    
    train_set = CartoonFacesDataset(root=root, train=True)
    valid_set = CartoonFacesDataset(root=root, train=False)
    print(len(train_set))
    print(len(valid_set))

    train_set[0].show()
    valid_set[0].show()