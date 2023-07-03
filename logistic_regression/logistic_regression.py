import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_save_model(load_model=False):
    path = "../transactions.csv.zip"
    train_data = pd.read_csv(path)
    X_train = train_data.drop(columns="Class")
    y_train = train_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    if load_model:
        # Load the trained model from file
        model = joblib.load("./model.pkl")
    else:
        # Train the model
        model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
        model.fit(X_train, y_train)

        # Save the trained model to file
        joblib.dump(model, "./model.pkl")

    print(model.predict_proba(X_test))
    prediction = model.predict(X_test)
    print(prediction)
    # Print number of predictions, which are 1
    print(np.sum(prediction))
    print(model.score(X_test, y_test))

    # Print the number of frauds detected correctly
    num_frauds = np.sum(prediction * y_test)
    num_samples = len(y_test)

    num_non_frauds_test = len(y_test) - np.sum(y_test)
    num_frauds_test = np.sum(y_test)

    num_frauds_prediction = np.sum(prediction)
    num_non_frauds_prediction = len(prediction) - np.sum(prediction)

    num_non_frauds_correct = np.sum((1 - y_test) * (1 - prediction))
    num_non_frauds_incorrect = (len(prediction) - np.sum(prediction)) - (np.sum((1 - y_test) * (1 - prediction)))

    num_frauds_correct = np.sum(y_test * prediction)
    num_frauds_incorrect = np.sum(prediction) - num_frauds_correct

    print("Number of samples:", num_samples)
    print()
    print("Number of non-frauds in test set:", num_non_frauds_test)
    print("Number of frauds in test set:", num_frauds_test)
    print()
    print("Number of frauds in prediction:", num_frauds_prediction)
    print("Number of non-frauds in prediction:", num_non_frauds_prediction)
    print("Number of non-frauds detected correctly:", num_non_frauds_correct)
    print("Number of non-frauds detected incorrectly:", num_non_frauds_incorrect)
    print("Number of frauds detected correctly:", num_frauds_correct)
    print("Number of frauds detected incorrectly:", num_frauds_incorrect)

train_and_save_model(load_model=True)
