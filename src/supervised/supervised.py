import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import pickle

# DATA READING
clustered_data = pd.read_csv('..\..\data\labeled\\encoded.csv', encoding='unicode_escape')

# Select relevant features for model
relevant_data = clustered_data[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME', 'CLUSTER']]

# DATA SPLITTING
# Set feature and target data
X = relevant_data.drop(['CLUSTER'], axis=1)
y = relevant_data['CLUSTER']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL APPLYING
# Create the classifier
classifier = DecisionTreeClassifier()
# classifier = RandomForestClassifier(max_depth=5, n_estimators=41, random_state=5)
# classifier = GaussianNB()
# classifier = SVC(probability=True)

# Train the classifier on the training set
classifier.fit(X_train, y_train)

# Evaluate model on test set
score = classifier.score(X_test, y_test)

# Obtain predictions
# predicted = classifier.predict(X_test)
prediction_probas = classifier.predict_proba(X_test)
predicted = np.argmax(prediction_probas, axis=1)

# Compute precision, recall, F1-score, and ROC AUC
precision = precision_score(y_test, predicted, average='weighted', zero_division=1)
recall = recall_score(y_test, predicted, average='weighted', zero_division=1)
f1 = f1_score(y_test, predicted, average='weighted')
# roc_auc = roc_auc_score(y_test, predicted, average='weighted', multi_class='ovo')

# Print the results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
# print(f'ROC AUC: {roc_auc:.3f}')

# Save encoders to disk
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

