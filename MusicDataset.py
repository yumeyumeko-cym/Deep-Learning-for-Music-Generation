import numpy as np
from torch.utils.data import Dataset



class MusicDataset(Dataset):
    def __init__(self, songs, sequence_length):
        self.songs = songs
        self.sequence_length = sequence_length
        self.vocab_size = len(set(songs))

    def __len__(self):
        return len(self.songs) - self.sequence_length

    def __getitem__(self, idx):
        return (self.one_hot_encode(self.songs[idx:idx + self.sequence_length]),
                self.songs[idx + self.sequence_length])

    def one_hot_encode(self, sequence):
        sequence_encoded = np.zeros((self.sequence_length, self.vocab_size), dtype=np.float32)
        for i, symbol in enumerate(sequence):
            sequence_encoded[i, symbol] = 1.0
        return sequence_encoded