import os
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import numpy as np


# load data
metadata = pd.read_csv('COMP90086_2024_Project_train/train.csv')
image_dir = 'COMP90086_2024_Project_train/train'

class StackDataset:
    def __init__(self, metadata, image_dir, model, batch_size, num_epochs, learning_rate):
        self.metadata = metadata
        self.image_dir = image_dir
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Load the image data
        self.train_data, self.val_data = self.split_data()
        
        self.transformer = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            rescale=1./255
        )
        
        # Create the data loaders
        self.train_loader = self.create_loader(self.train_data, self.transformer)
        self.val_loader = self.create_loader(self.val_data, self.transformer)
        
        '''self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)'''
         
                
        def split_data(self):
            # Split the data into training and validation sets
            train_data = self.metadata.sample(frac=0.8, random_state=42)
            val_data = self.metadata.drop(train_data.index)
            return train_data, val_data
        
        
        
        
# Helper function to load images and resize them
def load_images(image_ids, image_dir, img_size=(299, 299)):  # InceptionV4 需要 299x299 输入
    images = []
    for img_id in image_ids:
        # 构建图像路径
        img_path = os.path.join(image_dir, f'{img_id}.jpg')  # 假设图像是 .jpg 格式
        # 加载并预处理图像
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)  # 调整图像大小
            img = img_to_array(img) / 255.0  # 归一化图像
            images.append(img)
        else:
            print(f"Image {img_id} not found.")
    return np.array(images)
