import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Task3_Images"

# Settings
IMAGE_SIZE = (227, 227)
TRAIN_SPLIT = 0.7
BATCH_SIZE = 10

def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_dataset():
    transform = get_transforms()
    datastore = datasets.ImageFolder(root=str(DATA_DIR), transform=transform)
    return datastore

def show_sample_images(datastore):
    class_names = datastore.classes

    plt.figure(figsize=(12, 7))
    j = 1

    for i in range(0, 75, 15):
        if i >= len(datastore):
            break

        image, label = datastore[i]
        ax = plt.subplot(1, 5, j)
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(class_names[label])
        ax.axis("off")
        j += 1

    plt.tight_layout()
    plt.show()

def create_dataloaders(datastore):
    total_len = len(datastore)
    train_len = int(TRAIN_SPLIT * total_len)
    val_len = total_len - train_len

    train_set, val_set = torch.utils.data.random_split(datastore, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_set, val_set, train_loader, val_loader