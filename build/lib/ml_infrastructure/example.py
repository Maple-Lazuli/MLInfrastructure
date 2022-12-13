import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from ml_infrastructure.model import Model
from ml_infrastructure.data_manager import DataManager
from ml_infrastructure.manager import Manager

if __name__ == "__main__":
    # Create a Network
    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # If the size is a square, you can specify with a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class SomeCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 2)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 2)
            self.conv3 = nn.Conv2d(64, 128, 2)
            self.fc1 = nn.Linear(128 * 3 * 3, 288)
            self.fc2 = nn.Linear(288, 288)
            self.fc3 = nn.Linear(288, 128)
            self.fc4 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x


    myLenet = Model(net=LeNet(), name="Example_Lenet", save_dir='./results', device=None, lr=0.001)
    myCNN = Model(net=SomeCNN(), name="Example_CNN", save_dir='./results', device=None, lr=0.001)

    # Create The Data Loader
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    location = './data'
    batch_size = 50

    train_set = torchvision.datasets.CIFAR10(root=location, train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=location, train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dm = DataManager(train_loader, test_loader, test_loader, classes)

    # Create The Manager

    manager = Manager(models=[myLenet, myCNN], data_manager=dm, epochs=10)

    print("Starting Training and Evaluation")
    manager.perform()

    print("Saving Results")
    manager.save_watcher_results(save_location='./results', save_name='Example.json')

    print("Shutting Down")
    manager.shutdown_watcher()
