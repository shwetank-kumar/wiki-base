## Config

## File locations
dataset_file = "./data/wiki2/wiki2tokens.bin"
tokenizer_path = "./data/wiki2/wiki2tokenizer"                
checkpoint_dir = "./models/checkpoints/wiki2"
model_weight_file = "weights.pkl"


## Network params
block_size = 128
# batch_size = 32
emb_dim = 64
num_layers = 4
num_heads = 16
dropout = 0.2

## Training parameters 
lr = 1e-3
eval_batch_size = 16