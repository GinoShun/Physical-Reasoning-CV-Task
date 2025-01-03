import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, ShiftScaleRotate, RGBShift, RandomBrightnessContrast, Perspective, HueSaturationValue, GaussNoise, CoarseDropout, Resize, ColorJitter
from albumentations.pytorch import ToTensorV2
import albumentations as A

import warnings
warnings.filterwarnings("ignore", message="h5py is running against HDF5")

class StackDataset(Dataset):
    def __init__(self, csv_file, image_dir, img_size, stable_height, train=True, testMode=False, singleTask=True, doSplit=True):
        self.image_dir = image_dir
        self.train = train
        self.testMode = testMode
        self.singleTask = singleTask
        self.img_size = img_size
        self.doSplit = doSplit
        self.stable_height = stable_height
        self.metadata = pd.read_csv(csv_file)
        
        if not testMode:
            # Split data into training and validation sets
            if self.doSplit:
                self.train_data, self.val_data = self.split_data()
                self.data_frame = self.train_data if train else self.val_data
            else:
                self.data_frame = self.metadata

            
            self.image_ids = self.data_frame['id'].values
            # -1 if for classification
            self.labels = self.data_frame[self.stable_height].values - 1

            self.shapeset = self.data_frame['shapeset'].values - 1
            self.type = self.data_frame['type'].values - 1
            self.total_height = self.data_frame['total_height'].values - 2
            self.instability_type = self.data_frame['instability_type'].values
            self.cam_angle = self.data_frame['cam_angle'].values - 1       
        else:
            # test only
            self.data_frame = self.metadata
            self.image_ids = self.data_frame['id'].values
            self.labels = self.data_frame[self.stable_height].values

        # data augmentation
        if self.train:
            self.transform = Compose([
                Resize(self.img_size, self.img_size),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5), 
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                RandomBrightnessContrast(p=0.5),
                GaussNoise(var_limit=(10.0, 50.0), p=0.3),

                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = Compose([
                Resize(self.img_size, self.img_size),

                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def split_data(self):
        """Split data into train and validation sets."""
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_index, test_index in split.split(self.metadata, self.metadata[self.stable_height]):
            train_data = self.metadata.loc[train_index]
            val_data = self.metadata.loc[test_index]
        
        print(f"Training data: {len(train_data)} samples")
        print(f"Validation data: {len(val_data)} samples")
        return train_data, val_data

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f'{self.image_ids[idx]}.jpg')
        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        
        augmented = self.transform(image=image)
        image = augmented['image']

        if self.testMode:
            return image, self.image_ids[idx]
        else:
            if self.singleTask:
                label_main = self.labels[idx]
                return image, label_main
            else:
                label_main = self.labels[idx]
                label_shapeset = self.shapeset[idx]
                label_type = self.type[idx]
                total_height = self.total_height[idx]
                label_instability = self.instability_type[idx]
                label_cam_angle = self.cam_angle[idx]
                return image, {
                    'stable_height': label_main,
                    'shapeset': label_shapeset,
                    'type': label_type,
                    'total_height': total_height,
                    'instability_type': label_instability,
                    'cam_angle': label_cam_angle
                }

if __name__ == '__main__':
    dataset = StackDataset(csv_file='data/train.csv', image_dir='data/train', img_size = 224, stable_height='stable_height', train=True)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
