import numpy as np
import torch

from dataclasses import dataclass
import requests


@dataclass
class Trainer:
    model: any = None
    data_manager: any = None
    ip: str = "0.0.0.0"
    port: int = 5000
    epochs: int = 1

    def train(self):
        train_loader = self.data_manager.train_loader
        val_loader = self.data_manager.validation_loader
        train_loss = []
        val_loss = []

        for epoch in range(self.epochs):
            loss = []
            for batch_idx, data_target in enumerate(train_loader):
                data = data_target[0]
                target = data_target[1]
                loss.append(self.model.step(data, target))
            epoch_loss = np.mean(loss)
            train_loss.append(epoch_loss)
            self.send_watcher('loss', 'training', epoch_loss, len(train_loss))

            with torch.no_grad():
                loss = []
                for data_target in val_loader:
                    data = data_target[0]
                    target = data_target[1]
                    loss.append(self.model.loss(data, target))
            epoch_loss = np.mean(loss)
            val_loss.append(epoch_loss)
            self.send_watcher('loss', 'validation', epoch_loss, len(val_loss))

    def send_watcher(self, metric, mode, value, idx):
        server = f'http://{self.ip}:{self.port}/'
        endpoint = ""

        if metric == 'loss':
            endpoint = "updateLoss"

        body = {
            'data': {
                'name': self.model.name,
                'mode': mode,
                'value': value,
                'index': idx
            }
        }

        url = server + endpoint
        requests.post(url, json=body)
