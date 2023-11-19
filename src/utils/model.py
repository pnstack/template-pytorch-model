# save model utils
import torch
import torch
import os


def save_model(model: torch.nn.Module, path: str) -> str:
    parent_folder = os.path.dirname(path)
    os.makedirs(parent_folder, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path

def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(path))
    return model