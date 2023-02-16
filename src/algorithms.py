import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import data_prep

def kmeans(df):

    data_prep.remove_nans(df)

    # MODEL BUILDING
    # Preparing data for model
    # Encode variables
    data_prep.encode(df)

    # Normalize the variables
    df_norm = data_prep.normalize(df, "zscore")

    # MODEL APPLYING
    # Specify the number of clusters
    k = 5
    # Initialize and fit KMeans model
    model = KMeans(n_clusters=k)
    model.fit(df_norm)
    # Get cluster assignments for each data point
    return model.labels_


