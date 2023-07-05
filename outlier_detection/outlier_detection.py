from sklearn.ensemble import IsolationForest
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# --- Use Voting Classifier to combine models with different subsets ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = "/data/mlproject22" if os.path.exists("/data/mlproject22") else ".."
train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
X_train = train_data.drop(columns = "Class")
y_train = train_data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

# Assuming X_train is your feature variables for training data

# Create an Isolation Forest model
isolation_forest = IsolationForest(contamination='auto', random_state=42)

# Fit the model on the training data
isolation_forest.fit(X_train)

# Predict the anomaly score for each data point
anomaly_scores = isolation_forest.decision_function(X_test)

# Specify a threshold to classify outliers
threshold = 0.05  # Adjust this threshold based on your dataset and requirements

# Identify outliers based on the anomaly scores
outliers = y_test[anomaly_scores < threshold]

# Print the identified outliers
print("num total:")
print(len(y_test))
print("num Outliers:")
print(len(outliers))
print("num frauds detected:")
print(sum(outliers))
print("num frauds in test set:")
print(sum(y_test))