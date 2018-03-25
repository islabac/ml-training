#! /usr/local/bin/python3

import os
import re
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

EXT = '.txt'
TEST_SIZE = 0.33
BASE_PATH = './data'


def read_txt_file(path):
    labels = []
    sentences = []
    for dir_path, dir_names, files in os.walk(path):
        for f_name in files:
            if f_name.lower().endswith(EXT):
                file = open(os.path.join(dir_path, f_name), 'r')
                for line in file:
                    sentence, label = re.split(r'\t+', line.rstrip('\t\n'))
                    sentences.append(sentence)
                    labels.append(label)
    return sentences, labels
    
X, y = read_txt_file(BASE_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)

clf = MultinomialNB()

clf.fit(X_train, y_train)

X_test = vectorizer.transform(X_test)
y_pred = clf.predict(X_test)

# Find accuracy score, confusion matrix, and classification report
print("Accuracy is ", accuracy_score(y_test, y_pred)*100)
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))
