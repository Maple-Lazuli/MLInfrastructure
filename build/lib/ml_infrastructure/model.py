import os
import socket
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, (2, 100))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, (2, 100))
        self.fc1 = nn.Linear(1194240, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model:
    def __init__(self, net=Net(), name="Model", save_dir="./results/", device=None, lr=0.001):
        self.creation_date = str(datetime.now())
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.net.to(device=self.device)
        self.name = name
        self.save_dir = os.path.join(save_dir, name)
        self.save_name = os.path.join(self.save_dir, f"{self.name}.pth")
        self.lr = lr
        self.verify_save_dir(os.path.join(save_dir, name))

    def step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def loss(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        return loss.item()

    def classify(self, inputs):
        return self.net(inputs.to(self.device))

    def save(self, mode=None):
        if mode is not None:
            save_name = os.path.join(self.save_dir, f"{self.name}-{mode}.pth")
            torch.save(self.net.state_dict(), save_name)
        else:
            torch.save(self.net.state_dict(), self.save_name)

    def load(self):
        self.net.load_state_dict(torch.load(self.save_name))

    def verify_save_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_model_info(self):
        return_dict = dict()
        for attribute, value in vars(self).items():
            return_dict[attribute] = str(value)

        return_dict['host'] = socket.getfqdn()

        return return_dict
