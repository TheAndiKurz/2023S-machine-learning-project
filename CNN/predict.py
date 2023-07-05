from sklearn.model_selection import train_test_split
from hyperparameters import *

import os
import pandas as pd
import numpy as np
from methods import Net, predict_pytorch

# from contracts import contract

import torch

net = Net(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

net.load_state_dict(torch.load("model.pth"))

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

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data).reshape(-1, 1)

output = predict_pytorch(net, np.array(x_test_data))

# test the accuracy
wrong_output_count = 0
frauds = 0
frauds_right_detected = 0
frauds_detected = 0
y_test_data = np.array(y_test_data)
for idx, o in enumerate(output):
    if y_test_data[idx] == 1:
        frauds += 1

    if o == y_test_data[idx] and o == 1:
        frauds_right_detected += 1

    if o == 1:
        frauds_detected += 1

    if o != y_test_data[idx]:
        wrong_output_count += 1

print(f"accuracy: {1 - wrong_output_count / len(output)}")

print(f"frauds: {frauds}")
print(f"frauds_detected: {frauds_detected}")
print(f"frauds_right_detected: {frauds_right_detected}")
print(f"frauds detected ratio: {frauds_right_detected / frauds}")
