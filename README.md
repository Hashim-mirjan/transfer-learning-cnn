# Fruit Image classifier using Transfer Learning (AlexNet, PyTorch)

This project implements an image classification pipeline using transfer learning with a pretrained convolutional neural network (AlexNet) in PyTorch.

## Overview

The goal of this project is to classify images of fruits into predefined categories by making use of a model that has already been trained on a large dataset (ImageNet). Instead of training a deep network from scratch, the model is adapted to a new task by modifying and retraining only part of the network.

This approach is particularly useful when working with smaller datasets, as it reduces training time and improves generalisation whilst maintaining the feature extraction from the pretrained CNN.

## Approach

The workflow consists of:

- Loading image data using
- Applying preprocessing and augmentation (resize, normalization, rotation) for generalization
- Splitting data into training and validation sets (70/30)
- Using a pretrained **AlexNet** model
- Replacing the final classification layer to match our dataset
- Training the model and evaluating on a validation set

## Transfer Learning Strategy

The pretrained convolutional layers are frozen, meaning their weights are not updated during training. These layers act as a feature extractor so they help to identify things like edges in the images.

## Results

The model performance is evaluated using:

- Training vs validation accuracy
- Confusion matrix

### Accuracy
![Accuracy Plot](outputs/accuracy_plot.png)

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)


## How to run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```
### 2) Run the project
```bash
python main.py
```
This will run the training and testing of the model. A preview of the input photos is shown, then the training starts. Once the model has been validated, the results will appear.

## Notes

The dataset is not included in this repository. To run the project, place your images in:

