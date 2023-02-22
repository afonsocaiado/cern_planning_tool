import pandas as pd

from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

import algorithms
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load data from CSV file
q1 = pd.read_csv('..\data\processed\\preprocessed_q1.csv', encoding='unicode_escape')

# Select relevant features for clustering
X1 = q1[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES']] # Simple important categorical values
X2 = q1[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PRIORITY_EN']] # All categorical
X3 = q1[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION']] # Fernando variables in presentation

# Apply algorithms
# model, labels = algorithms.kmeans(X1, 5, "minmax")
# labels = algorithms.dbscan(X1, 5, "minmax")

# Save the clustering model
# joblib.dump(model, 'models/clustering_model.joblib')

# Model evaluation
# silhouette = silhouette_score(X1, labels)
# print("\n Silhouette Coefficient", silhouette)

# calinski = calinski_harabasz_score(X1, labels)
# print("\n Calinski-Harabasz Index:", calinski)

silhouette_coeffs = []

for k in range(2,15):
    model, labels = algorithms.kmeans(X1, k, "minmax")
    silhouette = silhouette_score(X1, labels)
    silhouette_coeffs.append(silhouette)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 15), silhouette_coeffs)
plt.xticks(range(2, 15))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

# q1.to_csv('..\data\processed\clustered_q1.csv', index=False)