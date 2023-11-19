import torch
import torch.optim as optim
import torch.nn.functional as F
from src.models.model import ShapeClassifier

from src.configs.model_config import ModelConfig
from src.data.data_loader import train_loader, num_classes
from src.utils.logs import writer
from src.utils.train import train
from src.utils.test import test
import wandb
import json
wandb.init(project="template-pytorch-model", entity="nguyen")


def train_runner():

    config = ModelConfig().get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeClassifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    log_interval = 20
    # log models config to wandb
    wandb.config.update(config)

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        loss = train(train_loader, model=model, loss_fn=F.cross_entropy,
                     optimizer=optimizer)
        test(train_loader, model=model, loss_fn=F.cross_entropy)
        # 3. Log metrics over time to visualize performance
        wandb.log({"loss": loss})

        # save model
        torch.save(model.state_dict(), "model.pth")

        # 4. Log an artifact to W&B
        wandb.log_artifact("model.pth")
        # model.train()
