import torch
import torch.optim as optim
import torch.nn.functional as F
from src.models.model import ShapeClassifier

from src.configs.model_config import ModelConfig
from src.data.data_loader import train_loader, num_classes, val_loader
from src.utils.train import train
from src.utils.test import test
from src.utils.wandb import wandb
from src.utils.logs import logging
from src.utils.model import save_model


def main():

    config = ModelConfig().get_config()

    device = config.device

    model = ShapeClassifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # log models config to wandb
    wandb.config.update(config)

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        loss = train(train_loader, model=model, loss_fn=F.cross_entropy,
                     optimizer=optimizer)
        test(val_loader, model=model, loss_fn=F.cross_entropy)
        # 3. Log metrics over time to visualize performance
        wandb.log({"loss": loss})

        # save model
        save_model(model, "results/models/last.pth")

        # 4. Log an artifact to W&B
        # wandb.log_artifact("model.pth")
        # model.train()


if __name__ == "__main__":
    logging.info("Training model")
    main()
