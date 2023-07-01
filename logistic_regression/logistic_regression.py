import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pathlib
import torch

path = "/data/mlproject22" if os.path.exists("/data/mlproject22") else ".."
train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
X_train = train_data.drop(columns = "Class")
y_train = train_data["Class"]


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
model.fit(X_train, y_train)
print(model.predict_proba(X_test))
prediction = model.predict(X_test)
print(prediction)
# print number of predictions, which are 1
print(np.sum(prediction))
print(model.score(X_test, y_test))

# print the number of frauds detected correctly
num_frauds = np.sum(prediction * y_test)
print("Number of samples", len(y_test))
print()
print("Number of non-frauds in test set: ", len(y_test) - np.sum(y_test))
print("Number of frauds in test set: ", np.sum(y_test))
print()
print("Number of frauds in prediction: ", np.sum(prediction))
print("Number of non-frauds in prediction: ", len(prediction) - np.sum(prediction))
print("Number of non-frauds detected correctly: ", np.sum((1-y_test)*(1-prediction)))
print("Number of non-frauds detected incorrectly: ", np.sum(1 - prediction) - num_frauds)
print("Number of frauds detected correctly: ", num_frauds)
print("Number of frauds detected incorrectly: ", np.sum(prediction) - num_frauds)


