import pandas as pd
from sklearn.cluster import KMeans

# The Dataset Acquisition has been done before with data_acquisition.py

# Load data from CSV file
q1 = pd.read_csv('..\data\processed\\clean_q1.csv', encoding='unicode_escape')

print("Shape: ", q1.shape)

# DATA SPLITTING
# Split the DataFrame into a train set (70%) and a test set (30%)
# train, test = train_test_split(df, test_size=0.3, random_state=0)

# K MEANS
# Select relevant features for clustering
X1 = q1[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES']] # Simple important categorical values

# Normalize the features
X1_norm = (X1 - X1.mean()) / X1.std()

# Specify the number of clusters
k = 5

# Initialize and fit KMeans model
model = KMeans(n_clusters=k)
model.fit(X1_norm)

# Get cluster assignments for each data point
labels = model.labels_

# Add cluster labels to the original data
q1['cluster'] = labels

print("\n", q1.head())

# print("Train Shape: ", train.shape)
