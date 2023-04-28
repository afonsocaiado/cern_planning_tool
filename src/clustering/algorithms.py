from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from kmodes.kmodes import KModes

import utils
import gower

def kmeans(df, k, norm):

    # Normalize the variables
    df = utils.normalize(df, norm)
    
    # MODEL APPLYING
    # Initialize and fit KMeans model
    model = KMeans(n_clusters=k)
    model.fit(df)

    # Get cluster assignments for each data point
    return df, model

def dbscan(df, n, dist, norm):
    
    distance_matrix = 0

    if dist == "euclidean":

        # Normalize the variables
        df = utils.normalize(df, norm)

        # Perform clustering with DBSCAN
        model = DBSCAN(eps=0.3, min_samples=n)
        model.fit(df)

    elif dist == "gower":

        distance_matrix = gower.gower_matrix(df)

        # Perform clustering with DBSCAN
        model = DBSCAN(eps=0.3, min_samples=n, metric="precomputed")
        model.fit(distance_matrix)


    return df, model, distance_matrix

def kmodes(df, k, init_method):

    model = KModes(n_clusters=k, init = init_method, n_init = 5, verbose=1, cat_dissim=utils.jaccard_dissim_label)
    model.fit(df)

    return df, model