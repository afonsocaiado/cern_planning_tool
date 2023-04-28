from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np

from kmodes.kmodes import KModes

import algorithms
import utils

def silhouette_plots(initial_df, alg, cluster_range, enc, norm):
    for k in range(2,cluster_range):

        if alg == "kmeans":
            df, model = algorithms.kmodes(initial_df, k, norm)
        elif alg == "dbscan":
            df, model, distance_matrix = algorithms.dbscan(initial_df, k, "euclidean", enc, norm)

        silhouette = silhouette_score(initial_df, model.labels_, metric=utils.jaccard_dissim_silhouette)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(initial_df, model.labels_, metric=utils.jaccard_dissim_silhouette)

        # Plot the silhouette plot
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)

        # The 1st subplot is the silhouette plot
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(df) + (k + 1) * 10])

        y_lower = 10

        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[model.labels_ == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                0, ith_cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette plot for {} clusters".format(k))
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # Add a vertical line for the average silhouette score
        ax1.axvline(x=silhouette, color="red", linestyle="--")

        plt.show()

def silhouettes_scores(initial_df, alg, cluster_range, enc, norm):

    silhouette_coeffs = []

    for k in range(2,cluster_range):

        if alg == "kmeans":
            df, model = algorithms.kmeans(initial_df, k, norm)
        elif alg == "dbscan":
            df, model, distance_matrix = algorithms.dbscan(initial_df, k, "euclidean", enc, norm)

        silhouette = silhouette_score(df, model.labels_)
        silhouette_coeffs.append(silhouette)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, cluster_range), silhouette_coeffs)
    plt.xticks(range(2, cluster_range))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

def elbow_method(initial_df, cluster_range):
    
    # Elbow curve to find optimal K
    cost = []
    K = range(1,cluster_range)
    for num_clusters in list(K):
        model = KModes(n_clusters=num_clusters, init = "Huang", n_init = 5, verbose=1, cat_dissim=utils.jaccard_dissim_label)
        model.fit(initial_df)
        cost.append(model.cost_)
        
    plt.plot(K, cost, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()