import pandas as pd

from sklearn.metrics import silhouette_score

import algorithms

# Load data from CSV file
q1 = pd.read_csv('..\data\processed\\q1.csv', encoding='unicode_escape')

# Select relevant features for clustering
X1 = q1[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES']] # Simple important categorical values
X2 = q1[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PRIORITY_EN']] # All categorical
X3 = q1[['TITLE', 'GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PRIORITY_EN']]

# Apply algorithms
labels = algorithms.kmeans(X1, 5)

# # Add cluster labels to the data
# X1['CLUSTER'] = labels

# Model evaluation
silhouette_avg = silhouette_score(X1, labels)
print("\n", silhouette_avg)

# q1.to_csv('..\data\processed\clustered_q1.csv', index=False)