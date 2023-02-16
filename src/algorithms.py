import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import data_prep

def kmeans(df, k, norm):

    print(df.head())

    data_prep.remove_nans(df)

    print(df.head())

    # MODEL BUILDING
    # Preparing data for model
    # Encode variables
    data_prep.encode(df)

    print(df.head())

    # Normalize the variables
    df_norm = data_prep.normalize(df, norm)

    # MODEL APPLYING
    # Initialize and fit KMeans model
    model = KMeans(n_clusters=k)
    model.fit(df_norm)
    # Get cluster assignments for each data point
    return model.labels_


