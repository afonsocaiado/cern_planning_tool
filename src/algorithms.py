import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import data_prep

def kmeans(df, k, norm):

    data_prep.remove_nans(df)


    # MODEL BUILDING
    # Preparing data for model
    # Encode variables
    data_prep.encode(df)

    # Normalize the variables
    df_norm = data_prep.normalize(df, norm)

    # MODEL APPLYING
    # Initialize and fit KMeans model
    model = KMeans(n_clusters=k)
    model.fit(df_norm)
    # Get cluster assignments for each data point
    return model.labels_

def dbscan(df, n, norm):
    
    data_prep.remove_nans(df)


    # MODEL BUILDING
    # Preparing data for model
    # Encode variables
    data_prep.encode(df)

    # Normalize the variables
    df_norm = data_prep.normalize(df, norm)

    # Perform clustering with DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=n)
    cluster_labels = dbscan.fit_predict(df_norm)

    return cluster_labels
