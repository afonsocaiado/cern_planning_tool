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
    return data_prep.normalize(df, "zscore")