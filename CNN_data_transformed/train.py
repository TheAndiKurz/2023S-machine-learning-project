#!/usr/bin/env python
# coding: utf-8


# In[18]:


# Define hyperparameters
from hyperparameters import *

# In[2]:
from tqdm.notebook import tqdm, trange

import pandas as pd
import numpy as np
import os

# from contracts import contract

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from methods import (
    Net,
    oversample_data,
    undersample_data,
    train_and_evaluate_model,
    evaluate_saved_model,
)

from sklearn.model_selection import train_test_split


# Initialize the network
net = Net(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

filepath = os.path.dirname(os.path.abspath(__file__))
path = (
    "/data/mlproject22"
    if os.path.exists("/data/mlproject22")
    else filepath + "/../data/"
)
data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))

X = data.drop(columns=["Time", "Class"])
y = data["Class"]

x_train_data_norm, x_test_data, y_train_data_norm, y_test_data = train_test_split(
    X, y, test_size=0.25, shuffle=True
)

# oversample data training
print("oversampling data training...")
print(f"samples before: {len(x_train_data_norm)}")
x_train_data_oversampled, y_train_data_oversampled = oversample_data(
    x_train_data_norm, y_train_data_norm, fraud_ratio=FRAUD_RATIO
)
print(f"samples after: {len(x_train_data_oversampled)}")

x_train_data_oversampled = np.array(x_train_data_oversampled)
y_train_data_oversampled = np.array(y_train_data_oversampled).reshape(-1, 1)

train_and_evaluate_model(
    net,
    x_train_data_oversampled,
    y_train_data_oversampled,
    x_test_data,
    y_test_data,
    "oversampled",
)


# undersample data training
print("undersampling data training...")
print(f"samples before: {len(x_train_data_norm)}")
x_train_data_undersampled, y_train_data_undersampled = undersample_data(
    x_train_data_norm, y_train_data_norm, fraud_ratio=FRAUD_RATIO
)
print(f"samples after: {len(x_train_data_undersampled)}")


x_train_data_undersampled = np.array(x_train_data_undersampled)
y_train_data_undersampled = np.array(y_train_data_undersampled).reshape(-1, 1)

train_and_evaluate_model(
    net,
    x_train_data_undersampled,
    y_train_data_undersampled,
    x_test_data,
    y_test_data,
    "undersampled",
)

print("evaluating saved models...")
print("oversampled")
evaluate_saved_model(
    net,
    "oversampled",
    x_test_data,
    y_test_data,
)

print("undersampled")
evaluate_saved_model(
    net,
    "undersampled",
    x_test_data,
    y_test_data,
)
