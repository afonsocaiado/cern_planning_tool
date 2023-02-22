import pandas as pd

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import utils

def kmeans(df, k, norm):

    utils.remove_nans(df)

    # MODEL BUILDING
    # Preparing data for model
    # Encode variables
    utils.encode(df)

    # Normalize the variables
    df_norm = utils.normalize(df, norm)
    
    # MODEL APPLYING
    # Initialize and fit KMeans model
    model = KMeans(n_clusters=k)
    model.fit(df_norm)

    # Add cluster labels to the data
    df['CLUSTER'] = model.labels_

    # Get cluster assignments for each data point
    return model, model.labels_

def dbscan(df, n, norm):
    
    utils.remove_nans(df)

    # MODEL BUILDING
    # Preparing data for model
    # Encode variables
    utils.encode(df)

    # Normalize the variables
    df_norm = utils.normalize(df, norm)

    # Perform clustering with DBSCAN
    model = DBSCAN(eps=0.3, min_samples=n)
    cluster_labels = model.fit_predict(df_norm)

    # Add cluster labels to the data
    df['CLUSTER'] = cluster_labels

    return model, cluster_labels
