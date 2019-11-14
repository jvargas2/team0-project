import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
import pytorch_lightning as pl

class FeatureLinear(pl.LightningModule):
    def __init__(self, dataset, train_indices, test_indices, num_features, batch_size=None):
        super(FeatureLinear, self).__init__()
        self.dataset = dataset
        self.train_indices = train_indices
        self.test_indices = test_indices

        self.layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        tag_probabilities = self.layers(x)
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
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    @pl.data_loader
    def train_dataloader(self):
        batch_size = 500 if self.on_gpu else 200
        sampler = SubsetRandomSampler(self.train_indices)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        batch_size = 500 if self.on_gpu else 200
        sampler = SubsetRandomSampler(self.train_indices)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, sampler=sampler)
        return dataloader
