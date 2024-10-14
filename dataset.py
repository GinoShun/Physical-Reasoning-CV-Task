import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, ShiftScaleRotate, RGBShift, RandomBrightnessContrast, Perspective, HueSaturationValue, GaussNoise, CoarseDropout, Resize, ColorJitter
from albumentations.pytorch import ToTensorV2
import albumentations as A


class AddEdgeDetection(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(AddEdgeDetection, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # img 是 numpy 数组，形状为 (H, W, C)
        # 转换为灰度图
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 进行边缘检测
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        # 将边缘结果归一化到 [0, 1]
        edges = edges / 255.0
        # 将边缘结果添加为第 4 个通道
        edges = edges[:, :, np.newaxis]
        img_with_edges = np.concatenate((img, edges), axis=2)
        return img_with_edges

    def get_transform_init_args_names(self):
        return ()


class StackDataset(Dataset):
    def __init__(self, csv_file, image_dir, img_size, stable_height, train=True, testMode=False, remove6=False):
        self.image_dir = image_dir
        self.train = train
        self.testMode = testMode
        self.remove6 = remove6
        self.img_size = img_size
        self.stable_height = stable_height
        self.metadata = pd.read_csv(csv_file)
        
        if not testMode:
            # Split data into training and validation sets
            self.train_data, self.val_data = self.split_data()

            self.data_frame = self.train_data if train else self.val_data
            self.image_ids = self.data_frame['id'].values
            # -1 if for classification
            self.labels = self.data_frame[self.stable_height].values - 1

            self.shapeset = self.data_frame['shapeset'].values - 1
            self.type = self.data_frame['type'].values - 1
            self.total_height = self.data_frame['total_height'].values - 1
            self.instability_type = self.data_frame['instability_type'].values
            self.cam_angle = self.data_frame['cam_angle'].values - 1

            if self.remove6:
                # remove label 5 (original label 6)
                valid_indices = self.labels != 5
                self.labels = self.labels[valid_indices]
                self.image_ids = self.image_ids[valid_indices]

                self.shapeset = self.shapeset[valid_indices]
                self.type = self.type[valid_indices]
                self.total_height = self.total_height[valid_indices]
                self.instability_type = self.instability_type[valid_indices]
                self.cam_angle = self.cam_angle[valid_indices]

                if self.train:
                    print(f"Training data: {len(self.image_ids)} samples")
                else:
                    print(f"Validation data: {len(self.image_ids)} samples")         
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
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5), 
                # Perspective(scale=(0.05, 0.1), p=0.5),
                # RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                RandomBrightnessContrast(p=0.5),
                GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                # CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),

                AddEdgeDetection(p=1.0),  # Add edge detection
                Normalize(mean=(0.485, 0.456, 0.406, 0.0), std=(0.229, 0.224, 0.225, 1.0)),
                # Normalize(mean=0.0, std=1.0, max_pixel_value=255.0), 
                ToTensorV2(),
            ])
        else:
            self.transform = Compose([
                Resize(self.img_size, self.img_size),
                AddEdgeDetection(p=1.0),  # Add edge detection
                Normalize(mean=(0.485, 0.456, 0.406, 0.0), std=(0.229, 0.224, 0.225, 1.0)),
                # Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
                ToTensorV2(),
            ])

    def split_data(self):
        """Split data into train and validation sets."""
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_index, test_index in split.split(self.metadata, self.metadata[self.stable_height]):
            train_data = self.metadata.loc[train_index]
            val_data = self.metadata.loc[test_index]
        
        if not self.remove6:
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

        if self.testMode:
            return image, self.image_ids[idx]
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
                # 'total_height': total_height,
                'num_unstable': total_height - label_main,
                'instability_type': label_instability,
                'cam_angle': label_cam_angle
            }

if __name__ == '__main__':
    dataset = StackDataset(csv_file='data/train.csv', image_dir='data/train', img_size = 224, stable_height='stable_height', train=True)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
