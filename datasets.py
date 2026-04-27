import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# create class for ROSE dataset for torch later
class ROSE_Dataset(Dataset):
    def __init__(self, base_path, subsets=['SVC', 'DVC', 'SVC_DVC'], split='train', 
                 transform=None, target_size=None, transform_label=False, elastic_transform=None):
        # probs only gonna use SVC_DVC, that's superficial and deep vasculature combined
        # transforms if we're doing augmentations later
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        self.target_size = target_size
        self.transform_label = transform_label
        self.elastic_transform = elastic_transform

        for subset in subsets:
            img_dir = os.path.join(base_path, subset, split, 'img')
            mask_dir = os.path.join(base_path, subset, split, 'gt')

            for fname in sorted(os.listdir(img_dir)):
                if fname.endswith('.png'):
                    self.image_paths.append(os.path.join(img_dir, fname))
                    self.mask_paths.append(os.path.join(mask_dir, fname.replace('.png', '.tif')))

    def __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self, idx):
        image = np.array(cv2.cvtColor(cv2.imread((self.image_paths[idx])), cv2.COLOR_BGR2GRAY))
        mask = np.array(cv2.cvtColor(cv2.imread((self.mask_paths[idx])), cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        if self.target_size is not None:
            image = cv2.resize(
                image, 
                self.target_size,
                interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask, 
                self.target_size,
                interpolation=cv2.INTER_NEAREST)
            
        if self.elastic_transform:
                image, mask = self.elastic_transform(image, mask)
        if self.transform:
            image = self.transform(image)
            if (self.transform_label):
                mask = self.transform(mask)

        # (H, W) -> (1, H, W) float tensor normalized to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        return image, mask

# create class for OCTA500 3mm dataset
class OCTA5003M_Dataset(Dataset):
    def __init__(self, base_path, transform=None, target_size=None, 
                 transform_label=False, elastic_transform=None):
        # transforms if we're doing augmentations later
        self.transform = transform
        self.target_size = target_size
        self.image_paths = []
        self.mask_paths = []
        self.transform_label=transform_label
        self.elastic_transform = elastic_transform

        img_dir = os.path.join(base_path, 'Img/Projection_Maps/OCTA(ILM_OPL)')
        mask_dir = os.path.join(base_path, 'Labels/GT_Capillary')

        for fname in sorted(os.listdir(img_dir)):
            if fname.endswith('.bmp'):
                self.image_paths.append(os.path.join(img_dir, fname))
                self.mask_paths.append(os.path.join(mask_dir, fname))

    def __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self, idx):
        image = np.array(cv2.cvtColor(cv2.imread((self.image_paths[idx])), cv2.COLOR_BGR2GRAY))
        mask = np.array(cv2.cvtColor(cv2.imread((self.mask_paths[idx])), cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        if self.target_size is not None:
            image = cv2.resize(
                image, 
                self.target_size,
                interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(
                mask, 
                self.target_size,
                interpolation=cv2.INTER_NEAREST)
        
        if self.elastic_transform:
                image, mask = self.elastic_transform(image, mask)

        if self.transform:
            image = self.transform(image)
            if (self.transform_label):
                mask = self.transform(mask)
        

        # (H, W) -> (1, H, W) float tensor normalized to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        return image, mask