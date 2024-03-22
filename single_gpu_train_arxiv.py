##TODO: Read directly from disk - Convert this to a file for train, validation and metadata
##TODO: Add DDP support for multi GPU
##TODO: Add a scheduler for annealing of learning rate
##TODO: Ability to overwrite any argument from the command line
import pickle
import torch
import os
from models.wiki2 import Xformer_Scratch as Xformer
from torch.optim import Adam
from single_gpu_config import *
import sys

def load_train_objs():

    with open(dataset_file, "rb") as f:
            loaded_objects = pickle.load(f)

    # Unpack the tuple of objects
    vocab_size, tokenized_text, _ = loaded_objects
    
    if os.path.exists(checkpoint_path):
        # Load model from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
        
        # Initialize model
        model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout).to(device)
        
        # Load model state dict
        model.load_state_dict(model_state_dict)
    else:
        # If checkpoint does not exist, initialize new model
        model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout).to(device)

    dataset = tokenized_text
    optimizer = Adam(model.parameters(), lr=lr)
    return dataset, model, optimizer



class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 dataset: dict[str, tuple[torch.Tensor, torch.Tensor]],
                 optimizer: torch.optim.Optimizer,
                 device: str,
                 save_every: int,
                 batch_size: int,
                 ) -> None:
        self.device = device
        self.save_every = save_every
        # self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

    def _get_batch(self, dataset, block_size=256):
        idx = torch.randint(len(dataset) - block_size, (self.batch_size, ))
        x = torch.tensor([dataset[i:i+block_size] for i in idx], dtype=torch.long)
        y = torch.tensor([dataset[i+1:i+block_size+1] for i in idx], dtype=torch.long)
        return x.to(self.device), y.to(self.device)
    
    @torch.inference_mode()
    def _evaluate_loss(self, eval_batch, num_batches = 8):
        self.model.eval()
        loss_tr = []
        loss_vl = []
        for n in range(num_batches):
            xtr, ytr = eval_batch['train']
            xval, yval = eval_batch['validation']
            _, train_loss = self.model(xtr, ytr)
            _, val_loss = self.model(xval, yval)
            loss_tr.append(train_loss)
            loss_vl.append(val_loss)
        
        mean_train_loss = torch.tensor(loss_tr).mean().item()
        mean_val_loss = torch.tensor(loss_vl).mean().item()
        self.model.train()
        return mean_train_loss, mean_val_loss
    
    def _run_batch(self, xtr, ytr):
        # Forward pass
        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(xtr, ytr)        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
               
    def _run_epoch(self, epoch):
        losses = {"train": [], "validation": []}
        ## Main training loop
        xtr, ytr = self._get_batch(self.dataset['train'], block_size)
        print(xtr.shape, ytr.shape)
        sys.exit("Error message")
        xval, yval = self._get_batch(self.dataset['validation'], block_size)
        self._run_batch(xtr, ytr)

        ## Evaluate the losses and print
        tr_lossi, val_lossi = self._evaluate_loss({'train': (xtr, ytr), 'validation': (xval, yval)}, num_batches=eval_batch_size)
        losses["train"].append(tr_lossi)
        losses["validation"].append(val_lossi)
        ## Print losses
        if epoch % self.save_every == 0:
            print(epoch, ' --> train loss: ', tr_lossi, 'validation loss: ', val_lossi)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        checkpoint = {'model_state_dict': ckp}
        PATH = checkpoint_path
        torch.save(checkpoint, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def main(device, total_epochs, save_every, batch_size):
    dataset, model, optimizer = load_train_objs()
    trainer = Trainer(model, dataset, optimizer, device, save_every, batch_size)
    trainer.train(total_epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    ## Setup device
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
    main(device, args.total_epochs, args.save_every, args.batch_size)