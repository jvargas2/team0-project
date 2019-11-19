import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl

class CharacterBiLSTM(pl.LightningModule):
    def __init__(self, dataset, train_indices, test_indices, num_features=None, batch_size=None):
        super(CharacterBiLSTM, self).__init__()
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices

        self.hidden_size = 100

        self.character_embedding = nn.Embedding(
            num_embeddings=24,
            embedding_dim=100,
            padding_idx=23
        )

        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Linear(200, 4)

    def forward(self, x):
        character_embeddings = self.character_embedding(x)
        _, (hidden_output, _) = self.lstm(character_embeddings)
        features = hidden_output.transpose(0, 1)
        features = features.contiguous().view(-1, self.hidden_size * 2)
        tag_probabilities = self.linear(features)
        return tag_probabilities

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    @pl.data_loader
    def train_dataloader(self):
        batch_size = 30 if self.on_gpu else 5
        sampler = SubsetRandomSampler(self.train_indices)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        batch_size = 30 if self.on_gpu else 5
        sampler = SubsetRandomSampler(self.train_indices)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        return dataloader
