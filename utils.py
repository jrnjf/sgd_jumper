import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision import datasets, transforms, models
import threading
import csv


C10_MEAN, C10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def resnet18_cifar(num_classes=10):
    m = models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    return m


def cifar_loaders(dataset="cifar10", batch_size=128, root="./data"):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(C10_MEAN, C10_STD),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(C10_MEAN, C10_STD),
    ])

    if dataset.lower() == "cifar10":
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm_train)
        test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_test)
        num_classes = 10
    elif dataset.lower() == "cifar100":
        train_ds = datasets.CIFAR100(root=root, train=True, download=True, transform=tfm_train)
        test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=tfm_test)
        num_classes = 100
    else:
        raise ValueError("dataset must be 'cifar10' or 'cifar100'")

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_ld, test_ld, num_classes
def triangle_wave(t, T=1.0, max=1.0 , min=0.1):
    x = t % T
    if x < T/2:
        return min + (max - min) * (2 * x / T)
    else:
        return min + (max - min) * (2 - 2 * x / T)

class ScalarLogger(threading.Thread):
    def __init__(self, log_queue, file_path):
        super().__init__()
        self.log_queue = log_queue
        self.file_path = file_path
        self.daemon = True

    def run(self):
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Metadata columns + 100 parameter columns
            header = ['epoch', 'batch', 'lr'] + [f'p_{i}' for i in range(100)]
            writer.writerow(header)
            
            while True:
                data = self.log_queue.get()
                if data is None: break  # Poison pill to stop thread
                writer.writerow(data)
                self.log_queue.task_done()
