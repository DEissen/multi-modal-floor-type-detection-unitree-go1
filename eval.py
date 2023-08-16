import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime


def evaluate(model, ds_test, sensors, config):
    ds_test_loader = DataLoader(
        ds_test, batch_size=config["batch_size"], shuffle=True, drop_last=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in ds_test_loader:
            images, labels = data
            images = images[sensors[0]]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(ds_test)} test images: %d %%' % (
        100 * correct / total))
