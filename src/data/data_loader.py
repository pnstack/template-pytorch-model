from .dataset import CustomDataset
from torch.utils.data import DataLoader
from src.configs.model_config import ModelConfig
from .transform import data_transform
import os

num_classes = 3
config = ModelConfig().get_config()

train_dataset = CustomDataset(data_folder=os.path.join("data", 'raw'), transform=data_transform)

# # Calculate the split point
# split_index = int(0.8 * len(dataset))

# # Split the dataset into training and testing
# train_dataset = dataset[:split_index]
# test_dataset = dataset[split_index:]


train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

