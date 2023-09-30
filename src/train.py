import torch
import torch.optim as optim
import torch.nn.functional as F
from .models.model import ShapeClassifier

from src.configs.model_config import ModelConfig
from src.data.data_loader import train_loader, num_classes


def train():
    config = ModelConfig().get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeClassifier(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    log_interval = 20
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % log_interval == 0:
                current_loss = running_loss / log_interval
                print(
                    f"Epoch [{epoch + 1}/{config.epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {current_loss:.4f}")
                running_loss = 0.0

                # calculate the accuracy on the test set

                with torch.no_grad():
                    model.eval()
                    correct = 0
                    total = 0
                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        predicted = torch.argmax(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    print(f"Accuracy of the model on the test images: {100 * correct / total} %")
                # save the model
                torch.save(model.state_dict(), "model.pth")