import os

## Wiki2
project_name = "wiki103"
dataset_file = "./data/wiki103tokens.bin"
tokenizer_path = "./data/wiki103tokenizer"                
checkpoint_dir = "./models/checkpoints"
model_weight_file = "weights.pkl"

## Network params
block_size = 256
emb_dim = 384
num_layers = 16
num_heads = 8
dropout = 0.2

## Training parameters 
lr = 0.1e-3
eval_batch_size = 8
batch_size = 64

# Setup wandb config
import wandb

# Define a config dictionary object
wandb_config = {
    "block_size": block_size,
    "emb_dim": emb_dim,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "dropout": 0.2,
}
