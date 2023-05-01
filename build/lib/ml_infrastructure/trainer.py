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
    print_loss: bool = False

    def __post_init__(self):
        self.train_loss = list()
        self.val_loss = list()

    def train(self):
        train_loader = self.data_manager.train_loader
        val_loader = self.data_manager.validation_loader

        for epoch in range(self.epochs):
            loss = []
            for batch_idx, data_target in enumerate(train_loader):
                data = data_target[0]
                target = data_target[1]
                loss.append(self.model.step(data, target))
            epoch_loss = np.mean(loss)
            if self.print_loss:
                print(f"Mean Epoch Training Loss: {epoch_loss}")
            self.train_loss.append(epoch_loss)
            self.send_watcher('loss', 'training', epoch_loss, len(self.train_loss))
            with torch.no_grad():
                loss = []
                for data_target in val_loader:
                    data = data_target[0]
                    target = data_target[1]
                    loss.append(self.model.loss(data, target))
            epoch_loss = np.mean(loss)
            if self.print_loss:
                print(f"Mean Epoch Validaion Loss: {epoch_loss}")
            self.val_loss.append(epoch_loss)
            self.send_watcher('loss', 'validation', epoch_loss, len(self.val_loss))

        return {
            'train_loss': self.train_loss,
            'validation_loss': self.val_loss,
        }

    def send_watcher(self, metric, mode, value, idx):
        server = f'http://{self.ip}:{self.port}/'
        endpoint = ""
        if metric == 'loss':
            endpoint = "updateLoss"

        body = {
            'data': {
                'name': self.model.name,
                'mode': mode,
                'value': float(value),
                'index': idx
            }
        }

        url = server + endpoint
        requests.post(url, json=body)
