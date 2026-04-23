import torch.nn as nn
import torch.optim as optim
from torchvision import models

LEARNING_RATE = 0.0001

def build_alexnet(num_classes):
    alexnet = models.alexnet(pretrained=True)

    # Replace final layer
    alexnet.classifier[6] = nn.Linear(4096, num_classes)

    return alexnet

def get_loss_function():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=LEARNING_RATE)