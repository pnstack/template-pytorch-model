from .transform import data_transform
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.io import read_image


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

        if label == "circle":
            label = 0
        elif label == "square":
            label = 1
        elif label == "triangle":
            label = 2

        return image, label
    
