from src.data import load_dataset, show_sample_images, create_dataloaders
from src.model import build_alexnet, get_loss_function, get_optimizer
from src.train import train_and_validate_nn
from src.utils import plot_accuracy, plot_confusion_matrix

from torchsummary import summary

EPOCHS = 16

def main():
    datastore = load_dataset()
    class_names = datastore.classes

    show_sample_images(datastore)

    train_set, val_set, train_loader, val_loader = create_dataloaders(datastore)

    print("Validation set length:", len(val_set))

    model = build_alexnet(len(class_names))

    summary(model, (3, 227, 227))

    loss_fn = get_loss_function()
    optimizer = get_optimizer(model)

    train_acc, val_acc, preds, actuals = train_and_validate_nn(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        EPOCHS
    )

    plot_accuracy(train_acc, val_acc)
    plot_confusion_matrix(actuals, preds, class_names)


if __name__ == "__main__":
    main()