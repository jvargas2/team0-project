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
        elif features == 'aaindex':
            self.create_aaindex_features()
        elif features == 'aaindex2d':
            self.create_aaindex2d_features()
        elif features == 'aaindex-seqvec':
            self.create_aaindex_seqvec_features()
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

    def create_aaindex_features(self):
        training_df = self.read_csv('../data/training.csv')
        self.create_labels(training_df)
        aaindex_df = self.read_csv('../data/features/aaindex.csv', sep='\t')
        self.x = aaindex_df.values.tolist()
    
    def create_aaindex2d_features(self):
        training_df = self.read_csv('../data/training.csv')
        self.create_labels(training_df)
        self.x = training_df['protein'].values.tolist()

        self.alphabet = ['A', 'L', 'R', 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
        features = {}
        header = []

        for aa in self.alphabet:
            features[aa] = []

        i = 2
        aaindex = open("data/aaindex1", "r")
        for line in aaindex:            
            if line[0] == 'H':
                accession_num = line.split()[1]
                header.append(accession_num)
            elif line[0] == 'I':
                i = 0
                continue
            
            if i == 0:
                values = line.split()
                features['A'].append(float(values[0]))
                features['R'].append(float(values[1]))
                features['N'].append(float(values[2]))
                features['D'].append(float(values[3]))
                features['C'].append(float(values[4]))
                features['Q'].append(float(values[5]))
                features['E'].append(float(values[6]))
                features['G'].append(float(values[7]))
                features['H'].append(float(values[8]))
                features['I'].append(float(values[9]))
                i = 1
            elif i == 1:
                values = line.split()
                features['L'].append(float(values[0]))
                features['K'].append(float(values[1]))
                features['M'].append(float(values[2]))
                features['F'].append(float(values[3]))
                features['P'].append(float(values[4]))
                features['S'].append(float(values[5]))
                features['T'].append(float(values[6]))
                features['W'].append(float(values[7]))
                features['Y'].append(float(values[8]))
                features['V'].append(float(values[9]))
                i = 2
        
        self.aaindex_features = features
        self.header = header

    def get_aaindex2d(self, protein):
        feature_cnt = len(self.aaindex_features['A'])
        x = [torch.tensor(self.aaindex_features[aa]) if aa in self.alphabet else torch.zeros(feature_cnt) for aa in protein]
        protein_tensor = torch.stack(x, dim=0)
        return protein_tensor

    def create_aaindex_seqvec_features(self):
        df = self.read_csv('../data/features/aaindex_seqvec.csv', sep='\t')
        self.create_labels(df)
        df = df.drop(['Unnamed: 0','label', 'class'], axis=1)
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
        elif self.features == 'aaindex2d':
            protein_tensor = self.get_aaindex2d(self.x[idx])
            return protein_tensor, self.y[idx]
        else:
            return torch.tensor(self.x[idx], dtype=torch.float), self.y[idx]
