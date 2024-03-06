from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        Initialize the dataset with inputs and targets.
        
        :param inputs: Encoded and one-hot encoded sequences of inputs.
        :param targets: Targets for each sequence in inputs.
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Fetch the input-output pair at the specified index.
        
        :param idx: Index of the sample to retrieve.
        :return: A tuple containing the input and target for the specified index.
        """
        return self.inputs[idx], self.targets[idx]