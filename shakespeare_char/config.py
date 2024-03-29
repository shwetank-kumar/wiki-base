import os

## File paths
project_name = "shakespeare_char"
dataset_file = "./data/shakespeare_char_tokens.bin"
tokenizer_path = "./data/shakespeare_char_tokenizer"                
checkpoint_dir = "./models/checkpoints"
model_weight_file = "weights.pkl"

## Network params
block_size = 256
emb_dim = 384
num_layers = 6
num_heads = 6
dropout = 0.2

## Training parameters 
lr = 0.1e-3
eval_batch_size = 8
batch_size = 128

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
