import torch


def train_and_validate_nn(model, train_loader, val_loader, loss_function, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epoch_acc_log = [0]
    val_epoch_acc_log = [0]

    for epoch in range(epochs):
        print(f"Epoch number: {epoch + 1}")

        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            total += labels.size(0)
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epochs_loss = running_loss / len(train_loader)
        epoch_acc = 100 * running_correct / total
        epoch_acc_log.append(epoch_acc)

        print(
            "    - Training dataset. Got %d out of %d images correct (%.2f%%). Epoch loss: %.3f"
            % (running_correct, total, epoch_acc, epochs_loss)
        )

        model.eval()
        val_running_correct = 0.0
        total = 0

        predictions_log = []
        actuals = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                total += labels.size(0)

                val_outputs = model(images)
                _, predicted = torch.max(val_outputs.data, 1)

                predictions_log.extend(predicted.cpu().numpy())
                actuals.extend(labels.cpu().numpy())

                val_running_correct += (labels == predicted).sum().item()

        val_epoch_acc = 100 * val_running_correct / total
        val_epoch_acc_log.append(val_epoch_acc)

        print(
            "    - Validation dataset. Got %d out of %d images correct (%.2f%%)"
            % (val_running_correct, total, val_epoch_acc)
        )

    print("Finished")
    return epoch_acc_log, val_epoch_acc_log, predictions_log, actuals