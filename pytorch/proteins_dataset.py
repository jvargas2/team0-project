from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset

class ProteinsDataset(Dataset):
    def __init__(self, debug=False):
        characters = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

        character_to_index = {char:i for i, char in enumerate(characters)}
        label_to_index = {'DNA': 0, 'RNA': 1, 'DRNA': 2, 'nonDRNA': 3}

        cwd = path.dirname(path.abspath(__file__))
        data_path = cwd + '/../data/training.csv'
        df = pd.read_csv(data_path)

        if debug:
            df = df.sample(10)

        df['label'] = df.apply(lambda row: label_to_index[row['class']], axis=1)
        proteins = list(df['protein'])

        self.y = list(df['label'])
        self.x = []
        
        for protein in proteins:
            indices = []

            for acid in protein:
                indices.append(character_to_index[acid])

            self.x.append(indices)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        max_length = 7176
        protein = self.x[idx]
        padding_length = 7176 - len(protein)
        padding = [0] * padding_length
        protein.extend(padding)
        return torch.tensor(protein), self.y[idx]
