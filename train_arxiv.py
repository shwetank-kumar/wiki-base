import torch
import os
import wandb
from models import Xformer_Scratch as Xformer
from torch.optim import Adam
import math
import pickle
from utils import evaluate_loss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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

# File paths
dataset_path = "./data/wiki2/wiki2tokens.bin"

def get_batch(dataset, device, block_size=256, batch_size=64):
    idx = torch.randint(len(dataset) - block_size, (batch_size, ))
    x = torch.tensor([dataset[i:i+block_size] for i in idx], dtype=torch.long)
    y = torch.tensor([dataset[i+1:i+block_size+1] for i in idx], dtype=torch.long)
    return x.to(device), y.to(device)

if __name__ == "__main__":
    # Initialize distributed backend
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    ## Load tokenized datasets
    with open(dataset_path, "rb") as f:
        loaded_objects = pickle.load(f)

    # Unpack the tuple of objects
    vocab_size, tokenized_text, tokens_dataset = loaded_objects

    # Initialize model and optimizer
    model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout).to(device)
    model = DDP(model, device_ids=[rank])

    optimizer = Adam(model.parameters(), lr=lr)
    
    # Create a learning rate scheduler for the run
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs - warmup_epochs, T_mult=1, eta_min=0)

    # Create loss lists
    losses = {"train": [], "validation": []}

    for epoch in range(n_epochs):
        # Adjust learning rate using scheduler
        scheduler.step(epoch)

        # Get batch for training and validation
        xtr, ytr = get_batch(tokenized_text['train'], device, block_size, batch_size)
        xval, yval = get_batch(tokenized_text['validation'], device, block_size, batch_size)
        
        # Forward pass
        logits, loss = model(xtr, ytr)
        
        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Evaluate loss
        tr_lossi, val_lossi = evaluate_loss(model, {'train': (xtr, ytr), 'validation': (xval, yval)}, num_batches=n_eval)
        losses["train"].append(tr_lossi)
        losses["validation"].append(val_lossi)
        
        ## Print losses
        if epoch % print_frequency == 0:
            print(epoch, ' --> train loss: ', tr_lossi, 'validation loss: ', val_lossi)

    # Clean up
    torch.distributed.destroy_process_group()
