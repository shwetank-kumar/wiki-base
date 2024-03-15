##TODO: Add multiGPU support in same node
##TODO: Add multiGPU support in differnt nodes

# Import the W&B Python Library
import wandb

# Start a W&B Run
run = wandb.init(
    project="train-gpt",
    notes="Futzing around",
)

wandb.config = {
    "block_size": 128,
    "batch_size": 32,
    "emb_dim": 64,
    "num_layers": 4,
    "num_heads": 16,
    "dropout": 0.2
}