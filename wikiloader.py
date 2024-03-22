import torch
from torch.utils.data import Dataset

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

class Wiki2Dataloader:
    def __init__(self, dataset, batch_size, block_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size
            batch_data = [self.dataset[idx] for idx in range(start_idx, end_idx)]
            x_batch = torch.stack([x for x, _ in batch_data]).to(self.device)
            y_batch = torch.stack([y for _, y in batch_data]).to(self.device)
            yield x_batch, y_batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


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
