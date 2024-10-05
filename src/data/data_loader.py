from .dataset import CustomDataset
from torch.utils.data import DataLoader
from src.configs.model_config import ModelConfig
from .transform import data_transform
import os
from torch.utils.data import random_split


num_classes = 3
config = ModelConfig().get_config()

all_dataset = CustomDataset(data_folder=config.data_dir, transform=data_transform)

# split to train, val, test
total_size = len(all_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    all_dataset, [train_size, val_size, test_size])


train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True)

val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=True)

test_loader = DataLoader(
    test_dataset, batch_size=config.batch_size, shuffle=True)


def get_train_dataset(batch_size):
    return DataLoader(
        all_dataset, batch_size=batch_size, shuffle=True)
