# Unsupervised Learning
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import mplcyberpunk
import numpy as np
import matplotlib.cm as cm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, required for 3D plotting
from sklearn.metrics import pairwise_distances
from jqmcvi.jqmcvi import base
import umap
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


def plot_elbow_curve(features, kmeans_kwargs: dict, n_clusters: int,
                     scaled_features: StandardScaler = StandardScaler()):
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
            plot_elbow_curve(scaled_features, kmeans_kwargs, n_clusters, scaled_features)
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
    plt.xlabel("Number of Clusters", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"})
    plt.ylabel("SSE", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"})
    plt.title("Elbow Curve for KMeans", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"})
    plt.grid(True)
    # Programmatically determine the point of the elbow:
    # If curve=”convex”, it will detect elbows
    # If the knee/elbow you are trying to identify is on a positive slope use direction=”increasing”
    kl = KneeLocator(range(1, n_clusters + 1), sse, curve="convex", direction="decreasing")
    print(f'Elbow point: {kl.elbow}\n')

    second_line = plt.axvline(x=kl.elbow, linewidth=3, color='blue', linestyle='--')
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

    return features, kmeans_kwargs, n_clusters, scaled_features


def calculate_silhouette_coefficients(scaled_features, kmeans_kwargs, n_clusters):
    """
        Example:
            silhouette_coefficients, optimal_k = calculate_silhouette_coefficients(scaled_features, kmeans_kwargs,
                                                                               n_clusters)
    """
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, n_clusters + 2):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_)
        silhouette_coefficients.append(score)
        # Get Cluster Labels
        print(f"Cluster labels {k}: {kmeans.labels_}")

    print("\nA list holds the silhouette coefficients for each k:\n"
          f"{silhouette_coefficients}")

    optimal_k = np.argmax(silhouette_coefficients) + 2  # Plus 2, since we start with k=2
    print(f'\nOptimal number of clusters: {optimal_k}')
    return silhouette_coefficients, optimal_k


def plot_silhouette_coefficients(silhouette_coefficients: list, optimal_k: int, n_clusters: int):
    """
        Example:
            plot_silhouette_coefficients(silhouette_coefficients, optimal_k, n_clusters)
    """
    plt.figure(figsize=(19.2, 10.8))
    first_line, = plt.plot(range(2, n_clusters + 2), silhouette_coefficients, marker='o',
                           linewidth=4, markersize=12,
                           markeredgewidth=4,
                           color='red', markerfacecolor='pink', markeredgecolor='darkgreen')
    second_line = plt.axvline(x=optimal_k, linewidth=3, color='violet', linestyle='--')
    mplcyberpunk.add_glow_effects()
    plt.xticks(range(2, n_clusters + 2))
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


def plot_silhouette(ax, n_clusters, selected_data):
    """
    Plot silhouette analysis on a given Axes object.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw the silhouette plot.
        n_clusters (int): The number of clusters to form.
        selected_data (DataFrame): Data used for performing clustering.

    Displays:
        Silhouette plot showing the silhouette coefficient for each sample in each cluster.
    """
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(selected_data)
    silhouette_avg = silhouette_score(selected_data, cluster_labels)
    sample_silhouette_values = silhouette_samples(selected_data, cluster_labels)

    print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = np.sort(sample_silhouette_values[cluster_labels == i])
        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")


def plot_clusters(ax, n_clusters, selected_data, markers):
    """
    Plot clusters on a given Axes object with jitter to improve visibility.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to draw the clusters.
        n_clusters (int): The number of clusters to form.
        selected_data (DataFrame): Data used for performing clustering.
        markers (list): List of marker styles to use for each cluster.

    Displays:
        A scatter plot of the data points in the dataset, jittered and colored by cluster.
    """
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(selected_data)
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    jitter_strength = 0.05
    marker_size = 600 / n_clusters

    for i in range(n_clusters):
        cluster_data = selected_data[cluster_labels == i]
        jitter_x = np.random.normal(0, jitter_strength, size=cluster_data.shape[0])
        jitter_y = np.random.normal(0, jitter_strength, size=cluster_data.shape[0])
        ax.scatter(cluster_data.iloc[:, 0] + jitter_x, cluster_data.iloc[:, 1] + jitter_y,
                   marker=markers[i], s=marker_size, lw=2, alpha=0.7,
                   c=colors[cluster_labels == i], edgecolor='k', label=f'Cluster {i}')

    legend = ax.legend(title="Cluster IDs", title_fontsize=14,
                       prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'},
                       labelcolor='white', loc='upper center', shadow=True, frameon=True, fancybox=True,
                       framealpha=0.8,
                       facecolor='darkgrey', edgecolor='red')
    # Legend Title Text Color Adjustment
    plt.setp(legend.get_title(), color='white')


def perform_clustering(range_n_clusters, selected_data, markers):
    """
    Perform clustering analysis and visualization for a range of cluster counts.

    Args:
        range_n_clusters (list of int): A list of integers representing the number of clusters to analyze.
        selected_data (DataFrame): The data on which to perform clustering.
        markers (list): List of marker styles for plotting clusters.

    Executes:
        This function creates and displays a series of subplots for each specified number of clusters,
        showing both silhouette analysis and cluster visualization.

    Example of use:
        Assuming you have a DataFrame `data` and you want to analyze clusters between 2 and 9
        markers = ['o', 's', '^', 'p', '*', 'd', 'v', '<', '>']
        perform_clustering(range(2, 10), data, markers)
    """
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(19.2, 10.8)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(selected_data) + (n_clusters + 1) * 10])

        plot_silhouette(ax1, n_clusters, selected_data)
        plot_clusters(ax2, n_clusters, selected_data, markers)

        plt.suptitle(f"Silhouette analysis for KMeans clustering on real data with n_clusters = {n_clusters}",
                     fontsize=14, fontweight='bold')
        plt.show()


def dunn_index(X, labels):
    """
    Calculate the Dunn index for the cluster labels provided for dataset X.
    """
    distances = pairwise_distances(X)
    unique_clusters = np.unique(labels)

    min_intercluster = np.inf
    max_intracluster = 0

    for i in unique_clusters:
        for j in unique_clusters:
            if i != j:
                intercluster_distances = distances[labels == i, :][:, labels == j]
                if intercluster_distances.size > 0:
                    min_intercluster = min(min_intercluster, np.min(intercluster_distances))

        intracluster_distances = distances[labels == i, :][:, labels == i]
        if intracluster_distances.size > 0:
            max_intracluster = max(max_intracluster, np.max(intracluster_distances))

    if max_intracluster == 0:
        return 0  # To avoid division by zero

    return min_intercluster / max_intracluster


def evaluate_clusters_and_plot(X, max_clusters):
    """
    Evaluates clustering for a range of cluster numbers and plots Dunn Index.
    """
    dunn_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        dunn_score = dunn_index(X, labels)
        dunn_scores.append(dunn_score)
        print(f"Dunn Index for k={k}: {dunn_score}\n")

    plt.figure(figsize=(19.2, 10.8))
    # Only set the label once
    for i, score in enumerate(dunn_scores):
        if i == 0:  # Set label only for the first data point
            plt.plot(range(2, max_clusters + 1)[i:i + 2], dunn_scores[i:i + 2], 'o-',
                     color='red', linewidth=4, markersize=12, markeredgewidth=4,
                     markerfacecolor='pink', markeredgecolor='darkgreen', label='Dunn Index per Cluster')
        else:
            plt.plot(range(2, max_clusters + 1)[i:i + 2], dunn_scores[i:i + 2], 'o-',
                     color='red', linewidth=4, markersize=12, markeredgewidth=4,
                     markerfacecolor='pink', markeredgecolor='darkgreen')

    plt.xlabel('Number of Clusters')
    plt.ylabel('Dunn Index')
    plt.title('Dunn Index for Various Numbers of Clusters')
    plt.legend(title="Optimization of Smart-Villages Clusters via Dunn Index", title_fontsize=14,
               prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'},
               loc='upper right', shadow=True, frameon=True, fancybox=True,
               framealpha=0.8, facecolor='darkgrey', edgecolor='yellow')
    mplcyberpunk.add_glow_effects()
    plt.grid(True)
    plt.show()


def evaluate_clusters_and_plot_snd_method(X, max_clusters):
    """
    Evaluates clustering for a range of cluster numbers and plots the Dunn Index using jqmcvi.
    """
    dunn_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        # Compute Dunn Index using jqmcvi
        dunn_score = base.dunn_fast(X, labels)
        dunn_scores.append(dunn_score)
        print(f"Dunn Index for k={k}: {dunn_score}\n")

    plt.figure(figsize=(19.2, 10.8))
    plt.plot(range(2, max_clusters + 1), dunn_scores, 'o-',
             color='red', linewidth=4, markersize=12, markeredgewidth=4,
             markerfacecolor='pink', markeredgecolor='darkgreen', label='Dunn Index per Cluster')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Dunn Index')
    plt.title('Dunn Index for Various Numbers of Clusters using jqmcvi')
    plt.legend(title="Optimization of Smart-Villages Clusters via Dunn Index", title_fontsize=14,
               prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'},
               loc='upper right', shadow=True, frameon=True, fancybox=True,
               framealpha=0.8, facecolor='darkgrey', edgecolor='yellow')
    plt.grid(True)
    mplcyberpunk.add_glow_effects()
    plt.show()


def create_pipelines(n_clusters=2):
    """
    Create preprocessing and clustering pipelines.

    Args:
    n_clusters (int): Number of clusters to use in KMeans.

    Returns:
    Tuple[Pipeline, Pipeline, Pipeline]: Returns preprocessing, clustering, and combined pipelines.
    """
    preprocessor = Pipeline([
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ])

    clusterer = Pipeline([
        ("kmeans", KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=100,
            max_iter=1000,
            random_state=42,
        )),
    ])

    combined_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ])

    return preprocessor, clusterer, combined_pipeline


def fit_pipeline(pipeline, data):
    """
    Fit the pipeline to the data and extract results.

    Args:
    pipeline (Pipeline): The combined preprocessing and clustering pipeline.
    data (DataFrame): The data to cluster.

    Returns:
    DataFrame: A DataFrame with principal components, cluster assignments, and centroids.
    """
    pipeline.fit(data)
    preprocessed_data = pipeline['preprocessor'].transform(data)
    predicted_labels = pipeline['clusterer']['kmeans'].labels_
    centroids = pipeline['clusterer']['kmeans'].cluster_centers_  # Getting centroids
    silhouette_avg = silhouette_score(preprocessed_data, predicted_labels)
    print(f"Silhouette Score: {silhouette_avg}")

    print(f"Centroids of the clusters: {centroids}")

    pcadf = pd.DataFrame(preprocessed_data, columns=["component_1", "component_2"])
    pcadf['predicted_cluster'] = predicted_labels

    # Return the DataFrame and centroids for later rendering
    return pcadf, centroids


def display_cluster_scatter_plot(data, centroids):
    """
    Plot the clustered data.

    Args:
    data (DataFrame): The DataFrame containing the principal components and cluster assignments.
    """
    plt.figure(figsize=(19.2, 10.8))
    scat = sns.scatterplot(
        x="component_1",
        y="component_2",
        s=50,
        data=data,
        hue="predicted_cluster",
        palette="Set2",
        legend="full"
    )

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids', marker='X')

    scat.set_title("Results of clustering of Smart-Villages data")
    legend = plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.0,
                        title="Cluster IDs", title_fontsize=14,
                        prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'},
                        labelcolor='white', shadow=True, frameon=True, fancybox=True,
                        framealpha=0.8,
                        facecolor='darkgrey', edgecolor='red'
                        )
    # Legend Title Text Color Adjustment
    plt.setp(legend.get_title(), color='white')
    plt.show()


def plot_explained_variance(data):
    """
    Plot the cumulative explained variance by the PCA.

    Args:
    data (DataFrame): The original dataset.

    """
    pca = PCA().fit(data)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print(f'Cumulative variance: {cumulative_variance}')

    plt.figure(figsize=(19.2, 10.8))
    line, = plt.plot(cumulative_variance, label='Cumulative Explained Variance', color='green')
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    legend = plt.legend(handles=[line], loc='best',
                        prop={'family': 'Times New Roman', 'size': 14, 'weight': 'bold'},
                        labelcolor='white', shadow=True, frameon=True, fancybox=True,
                        framealpha=0.8,
                        facecolor='darkgrey', edgecolor='red'
                        )
    # Legend Title Text Color Adjustment
    plt.setp(legend.get_title(), color='white')
    mplcyberpunk.add_glow_effects()
    plt.show()


def plot_3d_clusters(X, n_clusters, selected_columns):
    """
    Plots 3D scatter plots of clustered data with different combinations of features.

    Args:
    X (np.ndarray): The feature data as a numpy array.
    n_clusters (int): Number of clusters to use in KMeans.
    selected_columns (list): List of column names used for labeling the axes.

    Addtional information:
        bbox_to_anchor=(1, 1): Places the legend anchor point in the upper-right corner inside the axes or shape.
        bbox_to_anchor=(0.5, 0.5): Places the legend anchor in the center of the axes or shape.
        bbox_to_anchor=(1.05, 1): Shifts the legend slightly to the right of the upper-right corner
        of the axes, useful for placing the legend outside the main area of the chart.

    Example:
        df = pd.read_csv('data.csv')
        selected_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']
        X = df[selected_columns].values
        plot_3d_clusters(X, 9, selected_columns)
    """

    # Configure and perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Define the combinations of features to be plotted in each subplot
    combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    fig = plt.figure(figsize=(19.2, 10.8))

    for i, (x, y, z) in enumerate(combinations, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        sc = ax.scatter(X[:, x], X[:, y], X[:, z], c=labels.astype(float), cmap='jet', edgecolor='k')
        ax.set_xlabel(selected_columns[x])
        ax.set_ylabel(selected_columns[y])
        ax.set_zlabel(selected_columns[z])
        ax.set_title(f"Clusters {n_clusters} using {selected_columns[x]}, {selected_columns[y]}, {selected_columns[z]}")

    # Create legend for clusters
    colors = [sc.cmap(sc.norm(i)) for i in range(n_clusters)]
    legend_labels = [f'Cluster {i + 1}' for i in range(n_clusters)]
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                          markerfacecolor=colors[i], markersize=10) for i in range(n_clusters)]
    legend = plt.legend(handles=patches, bbox_to_anchor=(0.5, 0.5), loc='best',
                        prop={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'},
                        labelcolor='white', shadow=True, frameon=True, fancybox=True,
                        framealpha=0.8,
                        facecolor='darkgrey', edgecolor='red'
                        )
    # Legend Title Text Color Adjustment
    plt.setp(legend.get_title(), color='white')
    # Add a colorbar for all plots
    cbar = fig.colorbar(sc, ax=fig.axes, orientation='horizontal', shrink=0.8, pad=0.15)
    cbar.set_label('Cluster Label')

    # Adjust the layout of subplots within the figure
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.25)
    plt.show()


def apply_pca(scaled_data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Apply PCA to reduce the dimensionality of pre-scaled data.

    Args:
        scaled_data (np.ndarray): Pre-scaled data matrix.
        n_components (int): Number of principal components to return.

    Returns:
        np.ndarray: Data transformed into principal components.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    return principal_components


def cluster_visualization(data, n_clusters, markers):
    """
    Visualize clusters of data after applying KMeans.
    Args:
        data (ndarray): PCA-reduced data.
        n_clusters (int): Number of clusters for KMeans.
        markers (list): List of markers to use for each cluster.
    """
    if n_clusters > len(markers):
        print("Error: Not enough markers provided for the number of clusters.")
        return

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)

    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(data)

    # Color map for clusters
    colors = cm.nipy_spectral(np.linspace(0, 1, n_clusters))
    jitter_strength = 0.1  # Adjust this for better visibility

    for i in range(n_clusters):
        jitter_x = np.random.normal(0, jitter_strength, size=(data[cluster_labels == i, 0].size))
        jitter_y = np.random.normal(0, jitter_strength, size=(data[cluster_labels == i, 1].size))
        ax.scatter(data[cluster_labels == i, 0] + jitter_x, data[cluster_labels == i, 1] + jitter_y,
                   marker=markers[i], s=100, color=colors[i], label=f'Cluster {i + 1}')

    # Plotting centroids
    centers = clusterer.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax.scatter(c[0], c[1], marker='$%d$' % (i + 1), alpha=1, s=50, edgecolor='k')

    ax.set_title("Visualization of clustered data")
    ax.set_xlabel("Feature space for the 1st principal component")
    ax.set_ylabel("Feature space for the 2nd principal component")
    ax.legend(title="Cluster IDs")

    plt.show()


def display_cluster_umap(data, predicted_clusters, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Apply UMAP dimensionality reduction on high-dimensional data and plot the results.

    Args:
        data (ndarray): High-dimensional data to be reduced.
        predicted_clusters (array): Cluster labels for each point in data.
        n_neighbors (int): The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
        min_dist (float): The minimum distance apart that points are allowed to be in the low-dimensional representation.
        n_components (int): The number of dimensions to reduce to.

    Returns:
        None. Displays a scatter plot of the data reduced to two or three dimensions.
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(data)

    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=predicted_clusters,
        palette=sns.color_palette("hsv", len(set(predicted_clusters))),
        s=50,
        alpha=0.6,
        edgecolor='w'
    )
    scatter.set_title('UMAP Projection of the Clustered Data')
    legend = plt.legend(title='Cluster ID', loc='best',
                        prop={'family': 'Times New Roman', 'size': 12, 'weight': 'bold'},
                        labelcolor='white', shadow=True, frameon=True, fancybox=True,
                        framealpha=0.8,
                        facecolor='darkgrey', edgecolor='red'
                        )
    # Legend Title Text Color Adjustment
    plt.setp(legend.get_title(), color='white')
    plt.show()
