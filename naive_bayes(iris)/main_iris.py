#! /usr/local/bin/python3

from sklearn.externals import joblib
from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize static variable
TEST_SIZE = 0.33
MODEL_PATH = './model_iris.pkl'

# Download iris dataset
iris = load_iris()

# Create multinomail naive bayes classifier
clf = MultinomialNB()

# Set predictor and outcome value
X, y = iris.data, iris.target

# Split train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# Train model
clf.fit(X_train, y_train)

# Save model for further usage
s = joblib.dump(clf, MODEL_PATH)
print("Model saved successfully.")

# Test model
y_pred = clf.predict(X_test)

# Find accuracy score, confusion matrix, and classification report
print("Accuracy is ", accuracy_score(y_test, y_pred)*100)
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))
