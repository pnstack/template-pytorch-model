from src.train import train_runner
from src.auto import auto_hyper_parameter
import os
# set WANDB_API_KEY=$YOUR_API_KEY
# os.environ["WANDB_API_KEY"] = '7c0f2b9470a0a5c82bfae5bab4705344cb53288b'
# os.environ['WANDB_MODE'] = "offline"
if __name__ == "__main__":
    print("Training the model...")
    # train_runner()
    auto_hyper_parameter()
