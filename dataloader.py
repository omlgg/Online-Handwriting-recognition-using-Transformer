from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
from utils import VectorizeChar

class StrokeDataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing binary feature files.
            labels_df (Pandas Dataframe): Contains two columns file_path and transcript for mapping.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.vectorizer = VectorizeChar(257)
        self.data_dir = data_dir
        self.transform = transform

        # Load the labels file
        self.labels_df = labels_df
        self.file_paths = self.labels_df['file_path'].to_numpy()
        self.labels = self.labels_df['transcript'].to_numpy()

    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.vectorizer(self.labels[idx])

        # Load the binary feature file
        features = np.fromfile(file_path, dtype = np.float32) # Load the binary file

        # Convert features and label to tensors
        features = np.reshape(features, [-1, 20])
        #print(features.shape)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int32)

        if self.transform:
            features = self.transform(features)

        return features, label

def collate_fn(batch):
    src_batch = []
    tgt_batch = []
    src_mask = []
    tgt_mask = []
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)
        src_mask.append(torch.zeros(len(src)))
        tgt_mask.append(torch.zeros(len(tgt)))


    src_batch = pad_sequence(src_batch, batch_first = True)
    tgt_batch = pad_sequence(tgt_batch, batch_first = True)
    src_mask = pad_sequence(src_mask, batch_first = True, padding_value = float('-inf'))
    tgt_mask = pad_sequence(tgt_mask, batch_first = True, padding_value = float('-inf'))
    batch = {'src': src_batch, 'tgt': tgt_batch, 'src_msk': src_mask, 'tgt_msk': tgt_mask}
    return batch
