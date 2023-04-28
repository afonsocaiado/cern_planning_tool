import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import zscore
from kmodes.kmodes import KModes

import sys
sys.path.append('..')

import clustering_evaluation
import algorithms
import utils

import warnings
warnings.filterwarnings("ignore")

# DATA READING
joined_data = pd.read_csv('..\..\data\\tables\\activity.csv', encoding='unicode_escape')

# DATA PREPARATION
clean_data = utils.remove_nans(joined_data)
# clean_data = data_preprocessing.remove_outliers(clean_data)
processed_data = utils.prep_data(clean_data)

# MODEL BUILDING
# Select relevant features for clustering
# relevant_data = clean_data[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN']]
# relevant_data = clean_data[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'PRIORITY_EN']]
# relevant_data = clean_data[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'PRIORITY_EN']]
relevant_data = processed_data[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'CREATOR_NAME']]
# relevant_data = clean_data[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'PRIORITY_EN', 'CREATOR_NAME']]
# relevant_data = processed_data[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'PRIORITY_EN']]
# relevant_data = processed_data[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'PRIORITY_EN']]
# relevant_data = processed_data[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'PRIORITY_EN']]
# relevant_data = processed_data[['GROUP_RESPONSIBLE_NAME', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES']] # Simple important categorical values
# relevant_data = processed_data[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PRIORITY_EN']] # All categorical
# relevant_data = processed_data[['GROUP_RESPONSIBLE_NAME', 'RESPONSIBLE_WITH_DETAILS', 'ACTIVITY_TYPE_EN', 'WBS_NODE_CODE', 'FACILITY_NAMES', 'PREPARATION_DURATION', 'INSTALLATION_DURATION', 'COMMISSIONING_DURATION']] # Fernando variables in presentation

# PARAMETER TUNING DECISIONS
# clustering_evaluation.silhouettes_scores(relevant_data, "kmeans", 15, "label", "zscore") # 15 kmeans label encoder zscore is best?
clustering_evaluation.silhouette_plots(relevant_data, "kmeans", 15, "label", "Huang")
# clustering_evaluation.elbow_method(relevant_data, 20) 

# MODEL APPLYING
# df, model = algorithms.kmeans(relevant_data, 15, "zscore")
# df, model, distance_matrix = algorithms.dbscan(relevant_data, 15, "gower", "zscore")
df, model = algorithms.kmodes(relevant_data, 12, "Huang")

# # MODEL EVALUATION
# silhouette = silhouette_score(df, model.labels_)
# print("\n Silhouette Coefficient", silhouette)
silhouette = silhouette_score(relevant_data, model.labels_, metric=utils.jaccard_dissim_silhouette)
print("\n Silhouette Coefficient", silhouette)
# calinski = calinski_harabasz_score(df, model.labels_)
# print("\n Calinski-Harabasz Index:", calinski)
jaccard_dissimilarity_matrix = utils.jaccard_dissim_matrix(relevant_data.values)
dbi = davies_bouldin_score(jaccard_dissimilarity_matrix, model.labels_)
print("Davies-Bouldin Index:", dbi)

# CREATE DATASET FOR SUPERVISED LEARNING
# joined_data['CLUSTER'] = model.labels_
# joined_data.to_csv('..\..\data\labeled\\raw.csv', index=False)
# clean_data['CLUSTER'] = model.labels_
# clean_data.to_csv('..\..\data\labeled\processed.csv', index=False)
# processed_data['CLUSTER'] = model.labels_
# processed_data.to_csv('..\..\data\labeled\encoded.csv', index=False)
