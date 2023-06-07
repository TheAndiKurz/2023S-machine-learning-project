import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import pathlib

path = "data/"
train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
X_train = train_data.drop(columns="Class")
y_train = train_data["Class"]

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

print(train_data)
1
