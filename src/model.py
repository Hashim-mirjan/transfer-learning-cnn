import torch.nn as nn
import torch.optim as optim
from torchvision import models

LEARNING_RATE = 0.0001

def build_alexnet(num_classes):
    alexnet = models.alexnet(pretrained=True)

    # Freeze convolutional feature extractor
    for param in alexnet.features.parameters():
        param.requires_grad = False

    # Replace final classifier layer
    alexnet.classifier[6] = nn.Linear(4096, num_classes)

    return alexnet

def get_loss_function():
    return nn.CrossEntropyLoss()

def get_optimizer(model):
    # Only optimize parameters that are trainable
    return optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )