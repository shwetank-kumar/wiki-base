import os

## File paths
## Wiki2
# dataset_file = "./data/wiki2/wiki2tokens.bin"
# tokenizer_path = "./data/wiki2/wiki2tokenizer"                
# checkpoint_dir = "./models/checkpoints/wiki2"
# model_weight_file = "weights.pkl"

# ## Shakespeare
# dataset_file = "./data/shakespeare/shakespearetokens.bin"
# tokenizer_path = "./data/shakespeare/shakespearetokenizer"                
# checkpoint_dir = "./models/checkpoints/shakespeare"
# model_weight_file = "weights.pkl"

## Shakespeare
project_name = "shakespeare_char"
dataset_file = "./data/shakespeare_char/shakespeare_char_tokens.bin"
tokenizer_path = "./data/shakespeare_char/shakespeare_char_tokenizer"                
checkpoint_dir = "./models/checkpoints/shakespeare_char"
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

# Create a SummaryWriter instance
tensorboard_dir = os.path.join(checkpoint_dir, 'tensorboard')

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
