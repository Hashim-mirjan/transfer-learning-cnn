import torch.nn as nn
import torch.optim as optim
from torchvision import models

def build_model(model_name, num_classes):
    if model_name == "alexnet":
        model = models.alexnet(pretrained=True)

        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(1024, num_classes)

    else:
        raise ValueError("Model not supported. Choose 'alexnet' or 'googlenet'.")

    return model

def get_loss_function():
    return nn.CrossEntropyLoss()

def get_optimizer(model, learning_rate):
    return optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )