import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

## Defaults 
tokenizer_path = "./data/wiki2/wiki2tokenizer"
train_file = './data/wiki2/train.bin'
val_file = './data/wiki2/val.bin'
block_size = 8
batch_size = 2
num_batches = 2
count = 0

class Wiki2(Dataset):
    def __init__(self, dataset_path: str, tokenizer_path: str = tokenizer_path, block_size: int = 8) -> Dataset:
        self.tokenizer_path = tokenizer_path
        self.dataset = np.memmap(dataset_path, mode='r')
        self.block_size = block_size
        self.length = len(self.dataset)

    ##TODO: Use collate function to pad the dataset while sampling instead of this
    def __len__(self):
        return (self.length  - self.block_size)
    
    def __getitem__(self, idx):  
        x = torch.tensor(self.dataset[idx : idx + self.block_size].copy(), dtype=torch.long)
        y = torch.tensor(self.dataset[idx + 1 : idx + self.block_size + 1].copy(), dtype=torch.long)
        return x, y

train = Wiki2(train_file, block_size=block_size)
val = Wiki2(val_file, block_size=block_size)
train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val, batch_size=batch_size, shuffle=True)

# for batch in train_dl:
#     # Process each batch
#     x, y = batch
#     print(x)
#     print(y)
#     count = count + 1
#     if count >= num_batches:
#         break

# x,y = next(iter(train_dl))
# print(x)
# print(y)
# x,y = next(iter(train_dl))
# print(x)
# print(y)