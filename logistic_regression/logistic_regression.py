import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# --- Use Voting Classifier to combine models with different subsets ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_model(load_model=False):
    path = "../transactions.csv.zip"
    train_data = pd.read_csv(path)
    X_train = train_data.drop(columns="Class")
    y_train = train_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    def undersample_data(X_train, y_train, random_state=0, fraud_ratio=0.02):
    # Combine X_train and y_train into a single dataframe
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    majority_class = train_data[train_data['Class'] == 0]
    minority_class = train_data[train_data['Class'] == 1]

    desired_non_frauds = int(len(minority_class) / fraud_ratio)

    # Undersample the majority class
    undersampled_majority = resample(majority_class,
                                    replace=False,  # Set to False for undersampling
                                    n_samples=desired_non_frauds,  # take 10 times as many non-fraudulent transactions
                                    random_state=random_state)

    # Combine the undersampled majority class with the minority class
    undersampled_data = pd.concat([undersampled_majority, minority_class])

    # Split back into features (X) and target (y)
    X_undersampled = undersampled_data.drop('Class', axis=1)
    y_undersampled = undersampled_data['Class']
    return X_undersampled, y_undersampled


def oversample_data(X_train, y_train, fraud_ratio=0.02):
    # Assuming X_train and y_train are your feature and target variables for training data

    # Apply SMOTE oversampling
    smote = SMOTE(random_state=42,sampling_strategy=fraud_ratio)
    X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)
    return X_oversampled, y_oversampled


def get_log_model(random_state=0, max_iter=1000):
    # Create and train a logistic regression model
    model = LogisticRegression(solver='liblinear', random_state=random_state, max_iter=max_iter)
    return model


# --- function to print accuracy metrics ---
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


# print csv header
print("model, fraud ratio, accuracy, non-frauds classified as non_fraud, non-frauds classified as fraud, frauds classified as fraud, frauds classified as non-fraud")

model = get_log_model()
model.fit(X_train, y_train)
print("logistic regression,0.00159,", end="")
print_accuracy(model, X_test, y_test, csv=True)

for fraud_ratio in [0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.5, 0.7, 1]:

    # --- Train model on oversampled data ---
    model_oversampled = get_log_model()
    X_oversampled, y_oversampled = oversample_data(X_train, y_train, fraud_ratio=fraud_ratio)
    model_oversampled.fit(X_oversampled, y_oversampled)
    print("oversampling,", fraud_ratio, ",", end="")
    print_accuracy(model_oversampled, X_test, y_test, csv=True)

    # --- Train model on undersampled data ---
    model_undersampled = get_log_model()
    X_undersampled, y_undersampled = undersample_data(X_train, y_train, fraud_ratio=fraud_ratio)
    model_undersampled.fit(X_undersampled, y_undersampled)
    print("undersampling,", fraud_ratio, ",", end="")
    print_accuracy(model_undersampled, X_test, y_test, csv=True)

if load_model:
        # Load the trained model from file
        model = joblib.load("./model.pkl")
else:
model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
model.fit(X_train, y_train)
print(model.predict_proba(X_test))
prediction = model.predict(X_test)
print(prediction)
# print number of predictions, which are 1
print(np.sum(prediction))
print(model.score(X_test, y_test))


train_and_save_model(load_model=True)
