from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset

class ProteinsDataset(Dataset):
    def __init__(self, debug=False, features='character'):
        self.features = features
        self.debug = debug
        
        if features == 'character':
            self.create_character_indices()
        elif features == 'protovec':
            self.create_protovec_features()
        elif features == 'acid':
            self.create_acid_features()

    def read_csv(self, file_path):
        cwd = path.dirname(path.abspath(__file__))
        data_path = cwd + '/' + file_path
        df = pd.read_csv(data_path)
        if self.debug:
            df = df.sample(10)
        return df

    def create_labels(self, df):
        label_to_index = {'DNA': 0, 'RNA': 1, 'DRNA': 2, 'nonDRNA': 3}

        df['label'] = df.apply(lambda row: label_to_index[row['class']], axis=1)
        self.y = list(df['label'])

    def create_character_indices(self):
        characters = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

        character_to_index = {char:i for i, char in enumerate(characters)}

        df = self.read_csv('../data/training.csv')
        self.create_labels(df)

        proteins = list(df['protein'])
        self.x = []
        
        for protein in proteins:
            indices = []

            for acid in protein:
                indices.append(character_to_index[acid])

            self.x.append(indices)

    def create_protovec_features(self):
        pass

    def create_acid_features(self):
        df = self.read_csv('../data/acid_features.csv')
        self.create_labels(df)
        df = df.get(['hydrophobicity', 'polarity'])
        # df = (df - df.mean()) / df.std()
        self.x = df.values.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.features == 'character':
            max_length = 7176
            protein = self.x[idx]
            padding_length = 7176 - len(protein)
            padding = [0] * padding_length
            protein.extend(padding)
            return torch.tensor(protein), self.y[idx]
        elif self.features == 'acid':
            return torch.tensor(self.x[idx], dtype=torch.float), self.y[idx]
