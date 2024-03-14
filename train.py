##TODO: Add multiGPU support in same node
##TODO: Add multiGPU support in differnt nodes

# Import the W&B Python Library
import wandb

# 1. Start a W&B Run
run = wandb.init(
    project="train-gpt",
    notes="Futzing around",
)

