# Handwritten-hindi-word-reconiser

## main.py
It contains the function predict which takes the input as an RGB image and returns a list
which contains the letters predicted.

## Model
(model.h5)
The model that was trained is saved as model.h5 which is then used in main.py file.

## Character set
(character.txt)
The character set on which the model has been trained.

## Requirements
(requirements.txt)
The required dependencies to run the model are mentioned in requirements.txt file.

## Images
This folder contains some sample images which the model was able to predict correctly.

## Data Processing
(Processing.ipynb, ModelTraing.ipynb)
Processing.ipynb PreProcesses and prepares the data(from the UCI machine learning repository on our character set)
into four numpy arrays which are then taken as input by the ModelTraing.ipynb which trains our model on a CNN.

## Dataset
* https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset
