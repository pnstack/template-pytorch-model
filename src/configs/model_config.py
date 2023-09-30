import torch

class ModelConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 20
    def get_config(self):
        return self