##TODO: Check if weights file from CUDA training works on MPS
##TODO: Write inference function
##TODO: Read directly from disk - Convert this to a file for train, validation and metadata
##TODO: Add DDP support for multi GPU using accelerate - makesure to calculate loss on the main process only
##TODO: Add a scheduler for annealing of learning rate
##TODO: Ability to overwrite any argument from the command line
##TODO: Activate pin memory and shuffle
import pickle
import torch
import os
from models.wiki2 import Xformer_Scratch as Xformer
from torch.optim import Adam
from train_config import *
from wikiloader import Wiki2Dataset, Wiki2Dataloader
from torch.utils.data import Dataset
from accelerate import Accelerator

def load_train_objs():

    with open(dataset_file, "rb") as f:
            loaded_objects = pickle.load(f)

    # Unpack the tuple of objects
    vocab_size, tokenized_text, _ = loaded_objects
    
    if os.listdir(checkpoint_dir):
        # Initialize model
        model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout)
        # Load model state dict
        modelwgts = torch.load(os.path.join(checkpoint_dir, model_weight_file))
        # print(modelwgts)
        model.load_state_dict(modelwgts)

    else:
        # If checkpoint does not exist, initialize new model
        model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout)

    # dataset = tokenized_text
    train_dataset = Wiki2Dataset(tokenized_text['train'], block_size)
    val_dataset = Wiki2Dataset(tokenized_text['validation'], block_size)
    optimizer = Adam(model.parameters(), lr=lr)
    return train_dataset, val_dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, block_size: int):
    return Wiki2Dataloader(
        dataset,
        batch_size=batch_size,
        block_size=block_size,
        pin_memory=True,
        shuffle=True
    )


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 train_dataloader,
                 val_dataloader,
                 optimizer: torch.optim.Optimizer,
                 save_every: int,
                 batch_size: int,
                 ) -> None:
        self.save_every = save_every
        # self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
    
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
        accelerator.backward(loss)
        self.optimizer.step()
               
    def _run_epoch(self, epoch):
        losses = {"train": [], "validation": []}
        
        xtr, ytr = next(iter(self.train_dataloader))
        xval, yval = next(iter(self.val_dataloader))
        
        ## Train 1 batch
        self._run_batch(xtr, ytr)
        
        ## Evaluate the losses and print
        tr_lossi, val_lossi = self._evaluate_loss({'train': (xtr, ytr), 'validation': (xval, yval)}, num_batches=eval_batch_size)
        losses["train"].append(tr_lossi)
        losses["validation"].append(val_lossi)
        ## Print losses
        if epoch % self.save_every == 0:
            print('Epoch: ', epoch, ' | Train loss: ', tr_lossi, '| Validation loss: ', val_lossi)

    def _save_checkpoint(self):
        print(f"Training checkpoint saved at {checkpoint_dir}")
        # Unwrap
        model = accelerator.unwrap_model(self.model)
        state_dict = model.state_dict()
        # Use accelerator.save()
        accelerator.save(state_dict, os.path.join(checkpoint_dir, model_weight_file))


    def train(self, max_epochs: int):
        ## Main training loop
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint()

def main(total_epochs, save_every, batch_size):
    train_dataset, val_dataset, model, optimizer = load_train_objs()
    train_dataloader = prepare_dataloader(train_dataset, batch_size, block_size)
    val_dataloader = prepare_dataloader(val_dataset, batch_size, block_size)
    # model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, training_dataloader, scheduler)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, save_every, batch_size)
    trainer.train(total_epochs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    print("Device type being used by Hugging Face Accelerate:", device)
    main(args.total_epochs, args.save_every, batch_size)