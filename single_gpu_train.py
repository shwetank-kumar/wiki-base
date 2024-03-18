##TODO: Add multiGPU support in same node
##TODO: Add multiGPU support in differnt nodes

# Import the W&B Python Library
import torch
import wandb
from models import Xformer_Scratch as Xformer
from torch.optim import Adam
import math
import pickle
from utils import evaluate_loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datautils import Wiki2
from torch.utils.data import DataLoader
import os

# Setup device
# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # Fall back to CPU
        device = torch.device("cpu")

print("Device selected:", device)

# # Start a W&B Run
# run = wandb.init(
#     project="wiki2",
#     notes="Futzing around",
# )

# wandb.config = {
#     "block_size": 128,
#     "batch_size": 32,
#     "emb_dim": 64,
#     "num_layers": 4,
#     "num_heads": 16,
#     "dropout": 0.2,
# }

## Network params
block_size = 128
batch_size = 32
emb_dim = 64
num_layers = 4
num_heads = 16
dropout = 0.2

## Training params
lr = 0.001
warmup_epochs = 5
n_epochs = 200  # Adjust this according to your training duration

## Evaluation parameters
n_eval = 32
print_frequency = 10

# File paths
dataset_dir = os.path.dirname(__file__)
train_path = os.path.join(dataset_dir, 'data/wiki2/train.bin')
val_path = os.path.join(dataset_dir, 'data/wiki2/val.bin')
meta_path = os.path.join(dataset_dir, 'data/wiki2/meta.pkl')
tokenizer_path = os.path.join(dataset_dir, 'data/wiki2/wiki2tokenizer')

train = Wiki2(train_path, tokenizer_path, block_size)
val = Wiki2(val_path, tokenizer_path, block_size)
train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val, batch_size=batch_size, shuffle=True)


# def get_batch(dataset, device, block_size=256, batch_size=64):
#     idx = torch.randint(len(dataset) - block_size, (batch_size, ))
#     x = torch.stack([dataset[i:i+block_size] for i in idx])
#     y = torch.stack([dataset[i+1:i+block_size+1] for i in idx])

#     return x.to(device), y.to(device)

if __name__ == "__main__":

    ## Load tokenized datasets
    # with open(dataset_path, "rb") as f:
    #     loaded_objects = pickle.load(f)

    # Unpack the tuple of objects
    # vocab_size, tokenized_text, tokens_dataset = loaded_objects
    # tokenized_text.keys()
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    vocab_size = metadata["vocab_size"]


    # Import model and run a single pass 
    # xb, yb =  get_batch(tokenized_text['train'], device, block_size, batch_size)
    xb, yb = map(lambda t: t.to(device), next(iter(train_dl)))
    model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    logits, loss = model(xb,yb)
    xb.shape, yb.shape
    print('Measured loss:', loss.item())
    print('Expected loss:', -math.log(1./vocab_size))

    # Setting validation loss to a theoretical value from a uniform distirbution
    low_val_loss = -math.log(1./vocab_size)

    
    # Create a learning rate scheduler for the run
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs - warmup_epochs, T_mult=1, eta_min=0)

    # Create loss lists
    losses = {"train": [], "validation": []}

    for epoch in range(n_epochs):
        # Adjust learning rate using scheduler
        scheduler.step(epoch)
        # xtr, ytr = get_batch(tokenized_text['train'], device, block_size, batch_size)
        # xval, yval = get_batch(tokenized_text['validation'], device, block_size, batch_size)
        xtr, ytr = map(lambda t: t.to(device), next(iter(train_dl)))
        xval, yval =  map(lambda t: t.to(device), next(iter(val_dl)))
        eval_dataset = {'train': (xtr,ytr), 'validation': (xval, yval)}
        logits, loss = model(xtr,ytr)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tr_lossi, val_lossi = evaluate_loss(model, eval_dataset, num_batches=n_eval)
        losses["train"].append(tr_lossi)
        losses["validation"].append(val_lossi)

        ## Print losses
        if epoch % print_frequency == 0:
            print(epoch, ' --> train loss: ', tr_lossi, 'validation loss: ', val_lossi)

# ##TODO: Save parameters every run
# ##TODO: 