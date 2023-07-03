from sklearn.model_selection import train_test_split
from hyperparameters import *

import os
import pandas as pd
import numpy as np
from methods import Net, evaluate_saved_model

# from contracts import contract

import torch

# In[ ]:

filepath = os.path.dirname(os.path.abspath(__file__))
path = (
    "/data/mlproject22"
    if os.path.exists("/data/mlproject22")
    else filepath + "/../data/"
)
data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))

X = data.drop(columns=["Time", "Class"])
y = data["Class"]


x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(
    X, y, test_size=0.25, shuffle=True
)

net = Net(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

print("loading oversampled model...")
evaluate_saved_model(net, "oversampled", x_test_data, y_test_data)

print("loading undersampled model...")
evaluate_saved_model(net, "undersampled", x_test_data, y_test_data)
