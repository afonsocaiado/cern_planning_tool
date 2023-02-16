import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import data_prep
import algorithms

# Load data from CSV file
q1 = pd.read_csv('..\data\processed\\q1.csv', encoding='unicode_escape')

# K MEANS
# Select relevant features for clustering
X1 = q1[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES']] # Simple important categorical values
X2 = q1[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'PRIORITY_EN' , 'FACILITY_NAMES']] # All categorical

X1_norm = algorithms.kmeans(X1)

# MODEL APPLYING
# Specify the number of clusters
k = 5
# Initialize and fit KMeans model
model = KMeans(n_clusters=k)
model.fit(X1_norm)
# Get cluster assignments for each data point
labels = model.labels_
# Add cluster labels to the original data
X1['CLUSTER'] = labels

# MODEL EVALUATION
silhouette_avg = silhouette_score(X1, labels)
print("\n", silhouette_avg)

# q1.to_csv('..\data\processed\clustered_q1.csv', index=False)