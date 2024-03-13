import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        inputs: A tensor of shape (a, b, c) where a is the number of samples,
                b is the sequence length, and c is the one-hot vector dimension.
        targets: A tensor of shape (a,) where each element is the target for each sequence.
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the number of samples
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index. If your dataset is memory
        friendly, you could just return the data without the need to load them here.
        """
        return self.inputs[idx], self.targets[idx]