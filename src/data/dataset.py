from .transform import data_transform
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_files = os.listdir(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label =image_name[:len(image_name)-8]  # Extract the label from the filename
       

        image_path = os.path.join(self.data_folder, image_name)
        image = Image.open(image_path).convert("RGB")  # Ensure images are RGB

        if self.transform:
            image = self.transform(image)
        # print("label: ", label, image)
        if label == "circle":
            label = 0
        elif label == "square":
            label = 1
        elif label == "triangle":
            label = 2

        return image, label
    
# Create a CustomDataset class to represent the data
class StockDataset(Dataset):
    def __init__(self, data):
        self.data =  data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx]
        labels = features['Close']  # Target is the next day's closing price
        return features, labels