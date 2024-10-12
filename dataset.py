import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, ShiftScaleRotate, RGBShift, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2

class StackDataset(Dataset):
    def __init__(self, csv_file, image_dir, img_size, stratify_column, train=True):
        self.image_dir = image_dir
        self.train = train
        self.img_size = img_size
        self.stratify_column = stratify_column
        self.metadata = pd.read_csv(csv_file)  # 加载 CSV 文件
        
        # Split data into training and validation sets
        self.train_data, self.val_data = self.split_data()

        # 根据train标志选择是训练集还是验证集
        self.data_frame = self.train_data if train else self.val_data
        self.image_ids = self.data_frame['id'].values
        self.labels = self.data_frame[self.stratify_column].values

        # data augmentation
        if self.train:
            self.transform = Compose([
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                RandomBrightnessContrast(p=0.5),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # 验证集/测试集仅应用基本的转换和归一化
            self.transform = Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def split_data(self):
        """Split data into train and validation sets."""
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(self.metadata, self.metadata[self.stratify_column]):
            train_data = self.metadata.loc[train_index]
            val_data = self.metadata.loc[test_index]
        print(f"Training data: {len(train_data)} samples")
        print(f"Validation data: {len(val_data)} samples")
        return train_data, val_data

    def __len__(self):
        # 返回数据集大小
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 获取图像路径
        img_name = os.path.join(self.image_dir, f'{self.image_ids[idx]}.jpg')
        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        
        # 应用增强
        augmented = self.transform(image=image)
        image = augmented['image']

        # 获取标签
        label = self.labels[idx]
        
        return image, label

if __name__ == '__main__':
    dataset = StackDataset(csv_file='data/train.csv', image_dir='data/train_images', img_size = (299, 299), stratify_column='label', train=True)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
