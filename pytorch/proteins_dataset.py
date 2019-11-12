from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset

class ProteinsDataset(Dataset):
    def __init__(self, debug=False, features='character'):
        self.features = features
        self.debug = debug

        self.characters = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        if features == 'character' or features == 'onehot':
            self.create_character_indices()
        elif features == 'protvec':
            self.create_protovec_features()
        elif features == 'acid':
            self.create_acid_features()
        elif features == 'count':
            self.create_count_features()
        else:
            raise ValueError('Invalid features')

    def read_csv(self, file_path, sep=','):
        cwd = path.dirname(path.abspath(__file__))
        data_path = cwd + '/' + file_path
        df = pd.read_csv(data_path, sep=sep)
        if self.debug:
            df = df.sample(10)
        return df

    def create_labels(self, df):
        label_to_index = {'DNA': 0, 'RNA': 1, 'DRNA': 2, 'nonDRNA': 3}

        df['label'] = df.apply(lambda row: label_to_index[row['class']], axis=1)
        self.y = list(df['label'])

    def create_character_indices(self):
        character_to_index = {char:i for i, char in enumerate(self.characters)}

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
        training_df = self.read_csv('../data/training.csv')
        self.create_labels(training_df)
        protvec_df = self.read_csv('../data/features/protvec.csv', sep='\t')
        self.x = protvec_df.values.tolist()

    def create_acid_features(self):
        df = self.read_csv('../data/acid_features.csv')
        self.create_labels(df)
        df = df.get(['hydrophobicity', 'polarity'])
        # df = (df - df.mean()) / df.std()
        self.x = df.values.tolist()

    def create_count_features(self):
        df = self.read_csv('../AminoAcidcount/result.csv')
        self.create_labels(df)
        df = df.get(self.characters)
        self.x = df.values.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.features == 'character':
            max_length = 7176
            protein = self.x[idx]
            padding_length = 7176 - len(protein)
            padding = [23] * padding_length
            protein.extend(padding)
            return torch.tensor(protein), self.y[idx]
        elif self.features == 'onehot':
            max_length = 7176
            acid_indices = self.x[idx]
            padding_length = max_length - len(acid_indices)
            padding = [23] * padding_length
            acid_indices.extend(padding)
            acid_onehots = []

            for index in acid_indices:
                onehot = [0] * 23
                if index <= 22:
                    onehot[index] = 1
                acid_onehots.append(onehot)

            return torch.tensor(acid_onehots, dtype=torch.float), self.y[idx]
        else:
            return torch.tensor(self.x[idx], dtype=torch.float), self.y[idx]
