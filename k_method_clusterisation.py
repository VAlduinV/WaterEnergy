# Unsupervised Learning
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import mplcyberpunk
import numpy as np

plt.style.use('cyberpunk')


def load_data(file_path: str, columns: list):
    """
    Load data from Excel file and select specified columns.

    Example:
        Specify the list of columns you want to select
        columns_to_select = ['RD_m1_NEAR', 'RD_m2_NEAR', 'RD_m3_NEAR',
            'CITY2_NEAR', 'Kiyv_NEAR_', 'TOWN_NEAR_', 'Water_NEAR', 'occup_NEAR', 'power_NEAR']

            file_path = r'.\\data\\VILLAGE.xlsx'

        Call the load_data function to load the data and select the specified columns
            df, selected_data = load_data(file_path, columns_to_select)

    """
    df = pd.read_excel(file_path)
    selected_data = df[columns]
    return df, selected_data


def plot_elbow_curve(features, kmeans_kwargs: dict, n_clusters: int, scaled_features: StandardScaler = StandardScaler()):
    """
    Plot the elbow curve for selecting the optimal number of clusters.

    Example:
        Set the parameters
            kmeans_kwargs = {
                "init": "k-means++",
                "n_init": int,
                "max_iter": int,
                "random_state": int,
            }

        Call the function to build the elbow curve
            plot_elbow_curve(scaled_features, kmeans_kwargs, n_clusters)
    """

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, n_clusters + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        # The lowest SSE value (list)
        sse.append(kmeans.inertia_)
        # Final locations of the centroid
        print(f'Final locations of the centroid {k}-cluster:\n {kmeans.cluster_centers_}\n')
        # The number of iterations required to converge
        print(f'The number of iterations required to converge {k}-cluster: {kmeans.n_iter_}\n')
        # Get Cluster Labels
        print(f"Cluster labels {k}: {kmeans.labels_}\n")

    # ----------------------------------------------------------------------------------------------------------------#
    print(f"Statistics from starting initialization with the lowest SSE:\n {[i for i in enumerate(sse, start=1)]}\n")
    # ----------------------------------------------------------------------------------------------------------------#

    plt.figure(figsize=(19.2, 10.8))
    first_line, = plt.plot(range(1, n_clusters + 1), sse, marker='o', linewidth=4, markersize=12,
                           markeredgewidth=4,
                           color='red', markerfacecolor='blue', markeredgecolor='green')
    plt.xticks(range(1, n_clusters + 1))
    plt.xlabel("Number of Clusters", fontdict={"fontsize": 14,  "fontfamily": "Times New Roman"})
    plt.ylabel("SSE", fontdict={"fontsize": 14,  "fontfamily": "Times New Roman"})
    plt.title("Elbow Curve for KMeans", fontdict={"fontsize": 14,  "fontfamily": "Times New Roman"})
    plt.grid(True)
    # Programmatically determine the point of the elbow:
    # If curve=”convex”, it will detect elbows
    # If the knee/elbow you are trying to identify is on a positive slope use direction=”increasing”
    kl = KneeLocator(range(1, n_clusters + 1), sse, curve="convex", direction="decreasing")
    print(f'Elbow point: {kl.elbow}\n')

    second_line = plt.axvline(x=kl.elbow, linewidth=3, color='yellow', linestyle='--')
    mplcyberpunk.add_glow_effects()
    legend = plt.legend([first_line, second_line],
                        [f'Number of clusters: {n_clusters}', 'Optimal number of clusters: ({})'.format(kl.elbow)],
                        title='The Elbow Curve for KMeans', title_fontsize=14,
                        prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'},
                        labelcolor='white',
                        loc='upper right', shadow=True, frameon=True, fancybox=True, framealpha=0.8,
                        facecolor='black', edgecolor='red')
    # Legend Title Text Color Adjustment
    plt.setp(legend.get_title(), color='white')

    # The knee (or elbow) point is calculated simply by instantiating the KneeLocator class with
    # x, y and the appropriate curve and direction.
    # Here, kneedle.knee and/or kneedle.elbow store the point of maximum curvature.
    kl.plot_knee(figsize=(12.8, 10.8))
    mplcyberpunk.add_glow_effects()
    plt.show()


def calculate_silhouette_coefficients(scaled_features, kmeans_kwargs, n_clusters):
    """
        Example:
            silhouette_coefficients, optimal_k = calculate_silhouette_coefficients(scaled_features, kmeans_kwargs,
                                                                               n_clusters)
    """
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, n_clusters + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)
        # Get Cluster Labels
        print(f"Cluster labels {k}: {kmeans.labels_}")

    print("\nA list holds the silhouette coefficients for each k:\n"
          f"{[i for i in enumerate(silhouette_coefficients, start=1)]}")

    optimal_k = np.argmax(silhouette_coefficients) + 2  # Plus 2, since we start with k=2
    print(f'\nOptimal number of clusters: {optimal_k}')
    return silhouette_coefficients, optimal_k


def plot_silhouette_coefficients(silhouette_coefficients: list, optimal_k: int, n_clusters: int):
    """
        Example:
            plot_silhouette_coefficients(silhouette_coefficients, optimal_k, n_clusters)
    """
    plt.figure(figsize=(19.2, 10.8))
    first_line, = plt.plot(range(2, n_clusters + 1), silhouette_coefficients, marker='o',
                           linewidth=4, markersize=12,
                           markeredgewidth=4,
                           color='red', markerfacecolor='pink', markeredgecolor='darkgreen')
    second_line = plt.axvline(x=optimal_k, linewidth=3, color='violet', linestyle='--')
    mplcyberpunk.add_glow_effects()
    plt.xticks(range(2, n_clusters + 1))
    plt.xlabel("Number of Clusters", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"})
    plt.ylabel("Silhouette Coefficient", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"})
    plt.title("The silhouette coefficient", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"})
    plt.grid(True)
    legend = plt.legend([first_line, second_line],
                        [f'Number of clusters: {n_clusters}',
                         'Optimal number of clusters: ({})'.format(optimal_k)],
                        title='The silhouette coefficient', title_fontsize=14,
                        prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'},
                        labelcolor='white',
                        loc='upper right', shadow=True, frameon=True, fancybox=True, framealpha=0.8,
                        facecolor='black', edgecolor='red')
    # Legend Title Text Color Adjustment
    plt.setp(legend.get_title(), color='white')
    plt.show()


def cluster_analysis(true_labels, n_clusters: int,
                     scaled_features_rate: StandardScaler = StandardScaler()):
    """
        Example:
            cluster_analysis(true_labels, n_clusters, scaled_features_rate)
    """

    # Instantiate k-means and dbscan algorithms
    kmeans = KMeans(n_clusters=n_clusters)
    dbscan = DBSCAN(eps=0.3)

    # Fit the algorithms to the features
    kmeans.fit(scaled_features_rate)
    dbscan.fit(scaled_features_rate)

    # Compute the silhouette scores for each algorithm
    kmeans_silhouette = silhouette_score(scaled_features_rate, kmeans.labels_).round(2)
    dbscan_silhouette = silhouette_score(scaled_features_rate, dbscan.labels_).round(2)

    # Plot the data and cluster silhouette comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 10.8), sharex=True, sharey=True)
    fig.suptitle(f"Clustering Algorithm Comparison: Plots", fontsize=16)

    # Generating colors for each label dynamically for k-means
    unique_labels_kmeans = np.unique(kmeans.labels_)
    fte_colors_kmeans = {label: plt.cm.viridis(label / float(n_clusters - 1)) for label in unique_labels_kmeans}

    # Plotting for k-means
    for label in unique_labels_kmeans:
        cluster_indices = np.where(kmeans.labels_ == label)[0]
        ax1.scatter(scaled_features_rate[cluster_indices, 0], scaled_features_rate[cluster_indices, 1],
                    c=[fte_colors_kmeans[label]], label=f'Cluster {label}')

    ax1.set_title(f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 14})
    ax1.legend()

    # Generating colors for DBSCAN, including a default color for outliers (-1 label)
    unique_labels_dbscan = np.unique(dbscan.labels_)
    fte_colors_dbscan = {label: plt.cm.viridis(label / float(len(unique_labels_dbscan) - 1)) if label != -1 else 'gray' for label in unique_labels_dbscan}

    # Plotting for DBSCAN
    for label in unique_labels_dbscan:
        if label == -1:
            ax2.scatter(scaled_features_rate[dbscan.labels_ == -1, 0], scaled_features_rate[dbscan.labels_ == -1, 1],
                        c='gray', label='Outliers')
        else:
            cluster_indices = np.where(dbscan.labels_ == label)[0]
            ax2.scatter(scaled_features_rate[cluster_indices, 0], scaled_features_rate[cluster_indices, 1],
                        c=[fte_colors_dbscan[label]], label=f'Cluster {label}')

    ax2.set_title(f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 14})
    ax2.legend()

    plt.show()

    # Compute the adjusted Rand index scores for each algorithm
    ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
    ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)

    print(f"ARI score for k-means: {ari_kmeans:.4f}")
    print(f"ARI score for DBSCAN: {ari_dbscan:.4f}")

    return kmeans_silhouette, dbscan_silhouette, ari_kmeans, ari_dbscan
