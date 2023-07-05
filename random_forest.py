import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.model_selection import train_test_split

from under_oversample import oversample_data


def get_random_forest_model(random_state=0):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(criterion='gini', n_estimators=10, class_weight={0:10, 1:1}, random_state=random_state)
        return model


def print_accuracy(model, X_test, y_test, csv=False):
        prediction = model.predict(X_test)
        num_samples = len(y_test)
        num_frauds_in_test = np.sum(y_test)
        num_frauds_in_prediction = np.sum(prediction)
        num_frauds_detected_correctly = np.sum(y_test * prediction)
        num_non_frauds_detected_correctly = np.sum((1 - y_test) * (1 - prediction))

        if csv:
            print(model.score(X_test, y_test),
                num_non_frauds_detected_correctly, 
                num_samples - num_frauds_in_test - num_non_frauds_detected_correctly, 
                num_frauds_detected_correctly, 
                num_frauds_in_test - num_frauds_detected_correctly, 
                sep=",")
        else:
            print("Score: ", model.score(X_test, y_test))
            print()

            print("Number of samples", num_samples)
            print()
            print("Number of non-frauds in test set: ", num_samples - num_frauds_in_test)
            print("Number of frauds in test set: ", num_frauds_in_test)
            print()
            print("Number of frauds in prediction: ", num_frauds_in_prediction)
            print("Number of non-frauds in prediction: ", num_samples - num_frauds_in_prediction)
            print()
            print("Number of non-frauds detected correctly: ", num_non_frauds_detected_correctly)
            print("non-frauds detected as Fraud: ", num_samples - num_frauds_in_test - num_non_frauds_detected_correctly)
            print("Found Frauds: ", num_frauds_detected_correctly)
            print("Missed Frauds: ", num_frauds_in_test - num_frauds_detected_correctly)


def train_and_save_model(load_model=False):
    path = "../transactions.csv.zip"
    train_data = pd.read_csv(path)
    X_train = train_data.drop(columns="Class")
    y_train = train_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=None)

    if load_model:
        # Load the trained model from file
        model = joblib.load("./model.pkl")
    else:
        X_oversampled, y_oversampled = oversample_data(X_train, y_train, fraud_ratio=0.1)
        model = get_random_forest_model()
        model.fit(X_oversampled, y_oversampled)

        joblib.dump(model, "./model.pkl")
    
    print_accuracy(model, X_test, y_test)

train_and_save_model(load_model=False)



