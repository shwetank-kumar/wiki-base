##TODO: Update transformer sizing notebook
##TODO: Read directly from disk - Convert this to a file for train, validation and metadata
##TODO: Add DDP support for multi GPU using accelerate - makesure to calculate loss on the main process only
##TODO: Ability to overwrite any argument from the command line
##TODO: Activate pin memory and shuffle
##TODO: Batch size calculation and GPU memory utilization in an ipynb
import pickle
import torch
import os
from models.decoder import Xformer_Scratch as Xformer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ConstantLR
from config import *
from dataloaders import TokensDataset, TokensDataloader
from torch.utils.data import Dataset
from accelerate import Accelerator

def load_train_objs(total_epochs, warmup_epochs):

    with open(dataset_file, "rb") as f:
            loaded_objects = pickle.load(f)

    # Unpack the tuple of objects
    vocab_size, tokenized_text = loaded_objects
    
    if os.path.exists(os.path.join(checkpoint_dir, model_weight_file)):
        # Initialize model
        model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout)
        # Load model state dict
        modelwgts = torch.load(os.path.join(checkpoint_dir, model_weight_file), map_location=torch.device(accelerator.device.type))
        # print(modelwgts)
        model.load_state_dict(modelwgts)

    else:
        # If checkpoint does not exist, initialize new model
        model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout)

    # Calculate model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size} parameters")

    # dataset = tokenized_text
    train_dataset = TokensDataset(tokenized_text['train'], block_size)
    val_dataset = TokensDataset(tokenized_text['validation'], block_size)
    optimizer = Adam(model.parameters(), lr=lr)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_epochs - warmup_epochs, T_mult=1, eta_min=0)
    scheduler = ConstantLR(optimizer, lr, 1.0)
    return train_dataset, val_dataset, model, optimizer, scheduler

def prepare_dataloader(dataset: Dataset, batch_size: int, block_size: int):
    return TokensDataloader(
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
                 scheduler: torch.optim.lr_scheduler,
                 save_every: int,
                 batch_size: int,
                 ) -> None:
        self.save_every = save_every
        self.scheduler = scheduler
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
        self.scheduler.step()
               
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
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": tr_lossi,
                "val_loss": val_lossi,
            }
        )

        ## Print losses
        if epoch % self.save_every == 0:
            print('Epoch: ', epoch, 
                  ' | Train loss: ', tr_lossi, 
                  '| Validation loss: ', val_lossi, 
                  '| lr:', self.scheduler.get_last_lr()[0])
    

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


def main(total_epochs, save_every, warmup_epochs, batch_size):
    # Pass the config dictionary when you initialize W&B
    run = wandb.init(project=project_name, config=wandb_config)

    train_dataset, val_dataset, model, optimizer, scheduler = load_train_objs(total_epochs, warmup_epochs)
    train_dataloader = prepare_dataloader(train_dataset, batch_size, block_size)
    val_dataloader = prepare_dataloader(val_dataset, batch_size, block_size)
    # model, optimizer, training_dataloader, scheduler = accelerator.prepare(model, optimizer, training_dataloader, scheduler)
    model, optimizer, scheduler,train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, scheduler,train_dataloader, val_dataloader)
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, scheduler, save_every, batch_size)
    trainer.train(total_epochs)

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('warmup_epochs', type=int, help='Epochs used for warming up learning rate')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    print("Device type being used by Hugging Face Accelerate:", device)
    main(args.total_epochs, args.save_every, args.warmup_epochs, batch_size)