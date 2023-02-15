import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import encoding
import normalizing

# Load data from CSV file
q1 = pd.read_csv('..\data\processed\\clean_q1.csv', encoding='unicode_escape')

print("\n", q1.isna().sum())

# K MEANS
# Select relevant features for clustering
X1 = q1[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES']] # Simple important categorical values
# X2 = q1[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'PRIORITY_EN' , 'FACILITY_NAMES']] # All categorical

print("\n", X1)

# Encode variables
encoding.encode(X1)

# Normalize the variables
X1_norm = normalizing.normalize(X1, "zscore")

# Specify the number of clusters
k = 5

print("\n", X1_norm)

# Initialize and fit KMeans model
model = KMeans(n_clusters=k)
model.fit(X1_norm)

# Get cluster assignments for each data point
labels = model.labels_

# Add cluster labels to the original data
q1['CLUSTER'] = labels

print("\n", q1.head())

q1.to_csv('..\data\processed\clustered_q1.csv', index=False)

silhouette_avg_X1 = silhouette_score(X1, labels)

print("\n", silhouette_avg_X1)
