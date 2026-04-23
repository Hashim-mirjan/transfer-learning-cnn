import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

def plot_accuracy(train_accuracy, val_accuracy):
    epochs = range(0, len(train_accuracy))

    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "accuracy_plot.png")

    plt.show()

def plot_confusion_matrix(actuals, predictions, class_names):
    cm = confusion_matrix(actuals, predictions)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")

    plt.show()