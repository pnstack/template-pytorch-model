import wandb
from .logs import log_info
from src.configs.model_config import ModelConfig
config = ModelConfig().get_config()

if config.wandb:
    project = config.wandb_project
    entity = config.wandb_entity
    api_key = config.wandb_api_key
    wandb.login(key=api_key)
    wandb.init(project=project, entity=entity)
    log_info("Wandb is enabled")
else:
    log_info("Wandb is disabled")
    # disable wandb
    wandb.init(mode="disabled")
    pass
