import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl

class QuadLinear(pl.LightningModule):
    def __init__(self, dataset, train_indices, test_indices, num_features, batch_size=None):
        super(QuadLinear, self).__init__()
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.batch_size = 100
        self.hidden_size = 512
        self.learning_rate = .0001

        self.layers_dna = nn.Sequential(
            nn.Linear(num_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

        self.layers_rna = nn.Sequential(
            nn.Linear(num_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

        self.layers_drna = nn.Sequential(
            nn.Linear(num_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

        self.layers_nondrna = nn.Sequential(
            nn.Linear(num_features, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2)
        )

    def forward(self, x):
        dna = self.layers_dna(x).index_select(1, torch.tensor(0))
        rna = self.layers_rna(x).index_select(1, torch.tensor(0))
        drna = self.layers_drna(x).index_select(1, torch.tensor(0))
        nondrna = self.layers_nondrna(x).index_select(1, torch.tensor(0))
        tag_probabilities = torch.cat((dna, rna, drna, nondrna), 1)
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        batch_size = self.batch_size
        sampler = SubsetRandomSampler(self.train_indices)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        batch_size = self.batch_size
        sampler = SubsetRandomSampler(self.train_indices)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        return dataloader
