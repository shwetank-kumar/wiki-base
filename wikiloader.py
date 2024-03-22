import torch
from torch.utils.data import Dataset, DataLoader

class Wiki2Dataset(Dataset):
    def __init__(self, dataset, block_size):
        self.dataset = dataset
        self.block_size = block_size

    def __len__(self):
        return len(self.dataset) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.dataset[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.dataset[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y


class Wiki2Dataloader(DataLoader):
    def __init__(self, dataset, batch_size, block_size, *args, **kwargs):
       super().__init__(dataset, batch_size=batch_size, *args, **kwargs)
       self.block_size = block_size


    def __iter__(self):
        idx = torch.randint(len(self.dataset) - self.block_size, (len(self.dataset) // self.batch_size,))
        for i in idx:
            x, y = self.dataset[i]
            yield x, y


# Example usage:
# Assuming you have your dataset and block_size defined
# dataset = YourDatasetHere
# block_size = YourBlockSizeHere
# batch_size = YourBatchSizeHere
# device = YourDeviceHere

# custom_dataset = WikiDataset(dataset, block_size)
# custom_dataloader = Wiki2Dataloader(custom_dataset, batch_size, block_size, device)

# # You can iterate over custom_dataloader now to get batches
# for batch in custom_dataloader:
#     inputs, targets = batch
    # Your training code using inputs and targets