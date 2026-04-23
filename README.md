# Fruit Image classifier using Transfer Learning (AlexNet, PyTorch)

This project implements an image classification pipeline using transfer learning with a pretrained convolutional neural network (AlexNet) in PyTorch.

## Overview

The goal of this project is to classify images of fruits into predefined categories by making use of a model that has already been trained on a large dataset (ImageNet). Instead of training a deep network from scratch, the model is adapted to a new task by modifying and retraining only part of the network.

This approach is particularly useful when working with smaller datasets, as it reduces training time and improves generalisation whilst maintaining the feature extraction from the pretrained CNN.

## Approach

The workflow consists of:

- Loading image data
- Applying preprocessing and augmentation (resize, normalization, rotation) for generalization
- Splitting data into training and validation sets (70/30)
- Using a pretrained **AlexNet** model
- Replacing the final classification layer to match our dataset
- Training the model and evaluating on a validation set

This image preview shows what the training images look like. they have been normalized and rotated, hence the grainy, saturated appearance. This is key for more stable training and better generalization.
<img width="1281" height="400" alt="image" src="https://github.com/user-attachments/assets/045d3c01-fec6-4a3d-a321-ed4bbb467f38" />


## Transfer Learning Strategy

The pretrained convolutional layers are frozen, meaning their weights are not updated during training. These layers act as a feature extractor so they help to identify things like edges in the images.

## Model Comparison

To evaluate the effect of model architecture, both AlexNet and GoogLeNet were tested using the same training pipeline, including:

- identical preprocessing and augmentation
- the same train-validation split
- the same batch size and number of epochs
- the same optimizer

A baseline comparison was first carried out using the same learning rate for both models.

However, I observed that GoogLeNet required a higher learning rate to train effectively, which is consistent with deeper architectures being more sensitive to optimisation settings.

Therefore, I varied learning rates for GoogleNet and tested again for better convergence.

This allowed for both a controlled comparison between architectures and a more realistic evaluation of each model under appropriate training conditions  

## Results

The model performance is evaluated using:

- Training vs validation accuracy
- Confusion matrix

### Accuracy

<p align="center">
  <b>AlexNet (LR = 0.0001)</b>&nbsp;&nbsp;&nbsp;&nbsp;
  <b>GoogLeNet (LR = 0.0001)</b>&nbsp;&nbsp;&nbsp;&nbsp;
  <b>GoogLeNet (LR = 0.01)</b>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/869db750-e09c-46e5-b02c-e9232b8e3073" width="250"/>
  <img src="https://github.com/user-attachments/assets/740d880b-5172-4ac2-9eb7-9175d5e9f6c4" width="250"/>
  <img src="https://github.com/user-attachments/assets/7083ef52-96f3-4f17-83b8-3a0e5dcad6cf" width="250"/>
</p>

### Confusion Matrix

<p align="center">
  <img src="https://github.com/user-attachments/assets/291d1bb7-b034-4da4-93d5-8055fc211ded" width="250"/>
  <img src="https://github.com/user-attachments/assets/6194f9d5-03c0-4234-8c2f-8f96d4bbc16a" width="250"/>
  <img src="https://github.com/user-attachments/assets/763ea410-cfba-476f-a874-ee257ab4dbe0" width="250"/>
</p>

## Evaluation

Both AlexNet and GoogLeNet were able to successfully learn the classification task, achieving improving accuracy over the training epochs. The use of pretrained models allowed effective feature extraction even with a relatively small dataset, showing the potential if a much larger dataset was used.

Under identical training conditions, differences in performance were observed between the two architectures. AlexNet provided stable and consistent learning behaviour, while GoogLeNet initially showed slower convergence when using the same learning rate. After tuning the learning rate for GoogLeNet, performance improved. For general training stability the AlexNet model seems to be the stronger model here but both are very viable.

This approach is directly applicable to computer vision tasks in robotics, where pretrained CNNs are commonly used for perception tasks such as object recognition.

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

The dataset is not included in this repository.

