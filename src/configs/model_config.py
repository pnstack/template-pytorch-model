import torch

class ModelConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 5
        self.log_interval = 2 # Log every 2 batches => number of items is 32*2 = 64
    def get_config(self):
        return self