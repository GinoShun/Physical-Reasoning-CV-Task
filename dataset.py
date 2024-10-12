import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, ShiftScaleRotate, RGBShift, RandomBrightnessContrast, Perspective, HueSaturationValue, GaussNoise, CoarseDropout
from albumentations.pytorch import ToTensorV2

class StackDataset(Dataset):
    def __init__(self, csv_file, image_dir, img_size, stable_height, train=True, testMode=False):
        self.image_dir = image_dir
        self.train = train
        self.img_size = img_size
        self.stable_height = stable_height
        self.metadata = pd.read_csv(csv_file)
        
        if not testMode:
            # Split data into training and validation sets
            self.train_data, self.val_data = self.split_data()

            self.data_frame = self.train_data if train else self.val_data
            self.image_ids = self.data_frame['id'].values
            self.labels = self.data_frame[self.stable_height].values
        else:
            # test only
            self.data_frame = self.metadata
            self.image_ids = self.data_frame['id'].values
            self.labels = self.data_frame[self.stable_height].values

        # data augmentation
        if self.train:
            self.transform = Compose([
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.5), 
                Perspective(scale=(0.05, 0.1), p=0.5),
                RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                RandomBrightnessContrast(p=0.5),
                GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
                Normalize(mean=0.0, std=1.0, max_pixel_value=255.0), 
                ToTensorV2(),
            ])
        else:
            self.transform = Compose([
                # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
                ToTensorV2(),
            ])

    def split_data(self):
        """Split data into train and validation sets."""
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
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

        label = self.labels[idx]
        
        return image, label

if __name__ == '__main__':
    dataset = StackDataset(csv_file='data/train.csv', image_dir='data/train', img_size = (224, 224), stable_height='stable_height', train=True)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
