import pandas as pd
from sklearn.utils import resample


def undersample_data(X_train, y_train, random_state=0, fraud_ratio=0.02):
    # Combine X_train and y_train into a single dataframe
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    majority_class = train_data[train_data["Class"] == 0]
    minority_class = train_data[train_data["Class"] == 1]

    number_of_samples = len(train_data)

    desired_non_frauds = int(number_of_samples * (1 - fraud_ratio))

    # Undersample the majority class
    undersampled_majority = resample(
        majority_class,
        replace=False,  # Set to False for undersampling
        n_samples=desired_non_frauds,  # take 10 times as many non-fraudulent transactions
        random_state=random_state,
    )

    # Combine the undersampled majority class with the minority class
    undersampled_data = pd.concat([undersampled_majority, minority_class])

    # Split back into features (X) and target (y)
    X_undersampled = undersampled_data.drop("Class", axis=1)
    y_undersampled = undersampled_data["Class"]
    return X_undersampled, y_undersampled


def oversample_data(X_train, y_train, random_state=0, fraud_ratio=0.02):
    # Combine X_train and y_train into a single dataframe
    train_data = pd.concat([X_train, y_train], axis=1)

    # Separate majority and minority classes
    majority_class = train_data[train_data["Class"] == 0]
    minority_class = train_data[train_data["Class"] == 1]

    number_of_samples = len(train_data)

    desired_frauds = int(number_of_samples * fraud_ratio)

    # Undersample the majority class
    oversampled_minority = resample(
        minority_class,
        replace=True,  # Set to True for oversampling
        n_samples=desired_frauds,  # take 10 times as many non-fraudulent transactions
        random_state=random_state,
    )

    # Combine the undersampled majority class with the minority class
    oversampled_data = pd.concat([oversampled_minority, majority_class])

    # Split back into features (X) and target (y)
    X_oversampled = oversampled_data.drop("Class", axis=1)
    y_oversampled = oversampled_data["Class"]
    return X_oversampled, y_oversampled