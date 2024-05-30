# WaterEnergy/KMethod_cluster/k_method_clusterisation.py
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import numpy as np
import mplcyberpunk
import matplotlib.cm as cm  # Import cm for color mapping
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, required for 3D plotting
from sklearn.metrics import pairwise_distances
import umap

plt.style.use("cyberpunk")
fig_size = (13.6, 12.4)


def load_data(file_path: str, columns: list) -> tuple:
    """
    Load data from an Excel file and select specified columns.

    Args:
        file_path (str): Path to the Excel file.
        columns (list): List of columns to select.

    Returns:
        tuple: DataFrame with all data, DataFrame with selected columns.
    """
    df = pd.read_excel(file_path)  # Load data from Excel file
    selected_data = df[columns]  # Select specified columns
    return df, selected_data


def plot_elbow_curve(
    features: pd.DataFrame,
    kmeans_kwargs: dict,
    n_clusters: int,
    scaled_features: np.ndarray,
):
    """
    Plots the elbow curve to determine the optimal number of clusters.

    Args:
        features (pd.DataFrame): Original data features.
        kmeans_kwargs (dict): Arguments for KMeans.
        n_clusters (int): Maximum number of clusters to evaluate.
        scaled_features (np.ndarray): Scaled data features.

    Returns:
        int: Optimal number of clusters based on the elbow method.
    """
    sse = []  # List to hold sum of squared errors for each k
    for k in range(1, n_clusters + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)  # Initialize KMeans
        kmeans.fit(scaled_features)  # Fit KMeans
        sse.append(kmeans.inertia_)  # Append SSE to list

    plt.figure(figsize=fig_size)  # Create figure
    (first_line,) = plt.plot(
        range(1, n_clusters + 1),
        sse,
        marker="o",
        linewidth=4,
        markersize=12,
        markeredgewidth=4,
        color="red",
        markerfacecolor="blue",
        markeredgecolor="green",
    )  # Plot SSE values
    plt.xticks(range(1, n_clusters + 1))  # Set x-ticks
    plt.xlabel(
        "Number of Clusters", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"}
    )  # Set x-label
    plt.ylabel(
        "SSE", fontdict={"fontsize": 14, "fontfamily": "Times New Roman"}
    )  # Set y-label
    plt.title(
        "Elbow Curve for KMeans",
        fontdict={"fontsize": 14, "fontfamily": "Times New Roman"},
    )  # Set title
    plt.grid(True)  # Show grid

    kl = KneeLocator(
        range(1, n_clusters + 1), sse, curve="convex", direction="decreasing"
    )  # Find the elbow point
    second_line = plt.axvline(
        x=kl.elbow, linewidth=3, color="blue", linestyle="--"
    )  # Plot the elbow point
    mplcyberpunk.add_glow_effects()  # Add glow effects
    legend = plt.legend(
        [first_line, second_line],
        [
            f"Number of clusters: {n_clusters}",
            "Optimal number of clusters: ({})".format(kl.elbow),
        ],
        title="The Elbow Curve for KMeans",
        title_fontsize=14,
        prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
        labelcolor="black",
        loc="upper right",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="white",
        edgecolor="red",
    )  # Add legend
    plt.setp(legend.get_title(), color="black")  # Set legend title color
    plt.savefig("./data/data_photo/elbow_curve.pdf")  # Save figure
    plt.show()  # Show plot

    print(f"Optimal number of clusters according to the elbow method: {kl.elbow}")
    print(f"SSE for each k: {sse}")

    return kl.elbow


def calculate_silhouette_coefficients(
    scaled_features: np.ndarray, kmeans_kwargs: dict, n_clusters: int
) -> tuple:
    """
    Calculates the silhouette coefficients for a range of cluster numbers.

    Args:
        scaled_features (np.ndarray): Scaled data features.
        kmeans_kwargs (dict): Arguments for KMeans.
        n_clusters (int): Maximum number of clusters to evaluate.

    Returns:
        tuple: List of silhouette coefficients, optimal number of clusters.
    """
    silhouette_coefficients = []  # List to hold silhouette coefficients for each k
    for k in range(2, n_clusters + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)  # Initialize KMeans
        kmeans.fit(scaled_features)  # Fit KMeans
        score = silhouette_score(
            scaled_features, kmeans.labels_
        )  # Calculate silhouette score
        silhouette_coefficients.append(score)  # Append score to list

    optimal_k = (
        np.argmax(silhouette_coefficients) + 2
    )  # Find the optimal number of clusters

    print(f"Silhouette coefficients for each k: {silhouette_coefficients}")
    print(f"Optimal number of clusters according to silhouette method: {optimal_k}")

    return silhouette_coefficients, optimal_k


def plot_silhouette_coefficients(
    silhouette_coefficients: list, optimal_k: int, n_clusters: int
):
    """
    Plots the silhouette coefficients for a range of cluster numbers.

    Args:
        silhouette_coefficients (list): List of silhouette coefficients.
        optimal_k (int): Optimal number of clusters.
        n_clusters (int): Maximum number of clusters evaluated.
    """
    plt.figure(figsize=fig_size)  # Create figure
    (first_line,) = plt.plot(
        range(2, n_clusters + 1),
        silhouette_coefficients,
        marker="o",
        linewidth=4,
        markersize=12,
        markeredgewidth=4,
        color="red",
        markerfacecolor="pink",
        markeredgecolor="darkgreen",
    )  # Plot silhouette coefficients
    second_line = plt.axvline(
        x=optimal_k, linewidth=3, color="violet", linestyle="--"
    )  # Plot optimal number of clusters
    mplcyberpunk.add_glow_effects()  # Add glow effects
    plt.xticks(range(2, n_clusters + 1))  # Set x-ticks
    plt.xlabel("Number of Clusters")  # Set x-label
    plt.ylabel("Silhouette Coefficient")  # Set y-label
    plt.title("The silhouette coefficient")  # Set title
    plt.grid(True)  # Show grid
    legend = plt.legend(
        [first_line, second_line],
        [
            f"Number of clusters: {n_clusters}",
            "Optimal number of clusters: ({})".format(optimal_k),
        ],
        title="The silhouette coefficient",
        title_fontsize=14,
        prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
        labelcolor="black",
        loc="upper right",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="white",
        edgecolor="red",
    )  # Add legend
    plt.setp(legend.get_title(), color="black")  # Set legend title color
    plt.savefig("./data/data_photo/silhouette_coefficients.pdf")  # Save figure
    plt.show()  # Show plot


def plot_silhouette(ax, n_clusters: int, selected_data: pd.DataFrame):
    """
    Plots the silhouette analysis for KMeans clustering.

    Args:
        ax (matplotlib.axes.Axes): Axes object to draw the silhouette plot.
        n_clusters (int): Number of clusters.
        selected_data (pd.DataFrame): Data used for clustering.
    """
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize KMeans
    cluster_labels = clusterer.fit_predict(selected_data)  # Predict cluster labels
    silhouette_avg = silhouette_score(
        selected_data, cluster_labels
    )  # Calculate average silhouette score
    sample_silhouette_values = silhouette_samples(
        selected_data, cluster_labels
    )  # Calculate silhouette values for each sample

    print(f"Average silhouette score for n_clusters = {n_clusters}: {silhouette_avg}")

    y_lower = 10  # Initialize y_lower for silhouette plot
    for i in range(n_clusters):
        ith_cluster_silhouette_values = np.sort(
            sample_silhouette_values[cluster_labels == i]
        )  # Sort silhouette values for cluster i
        size_cluster_i = len(ith_cluster_silhouette_values)  # Get the size of cluster i
        y_upper = y_lower + size_cluster_i  # Calculate y_upper for cluster i
        color = cm.nipy_spectral(float(i) / n_clusters)  # Get color for cluster i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,  # Fill between y_lower and y_upper with silhouette values
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.text(
            -0.05, y_lower + 0.5 * size_cluster_i, str(i)
        )  # Add text label for cluster i
        y_lower = y_upper + 10  # Update y_lower for next cluster

    ax.set_title("The silhouette plot for the various clusters.")  # Set title
    ax.set_xlabel("The silhouette coefficient values")  # Set x-label
    ax.set_ylabel("Cluster label")  # Set y-label
    ax.axvline(
        x=silhouette_avg, color="red", linestyle="--"
    )  # Add vertical line for average silhouette score

    return silhouette_avg


def perform_clustering(
    range_n_clusters: list, selected_data: pd.DataFrame, markers: list
):
    """
    Performs clustering analysis and visualization for a range of cluster counts.

    Args:
        range_n_clusters (list): List of integers representing the number of clusters to analyze.
        selected_data (pd.DataFrame): The data on which to perform clustering.
        markers (list): List of marker styles for plotting clusters.

    Returns:
        list: Average silhouette scores for each number of clusters.
    """
    silhouette_avgs = []
    for n_clusters in range_n_clusters:
        fig, ax1 = plt.subplots()  # Create subplots
        fig.set_size_inches(19.2, 10.8)  # Set figure size
        ax1.set_xlim([-0.1, 1])  # Set x-axis limits
        ax1.set_ylim(
            [0, len(selected_data) + (n_clusters + 1) * 10]
        )  # Set y-axis limits

        silhouette_avg = plot_silhouette(
            ax1, n_clusters, selected_data
        )  # Plot silhouette analysis
        silhouette_avgs.append(silhouette_avg)
        plt.suptitle(
            f"Silhouette analysis for KMeans clustering on real data with n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )  # Set supertitle
        plt.show()  # Show plot

    return silhouette_avgs


def perform_clustering_ing(range_n_clusters, selected_data):
    """
    Perform clustering analysis and visualization for a specified range of cluster counts.

    Args:
        range_n_clusters (list): List of integers representing the number of clusters to analyze.
        selected_data (pd.DataFrame): Data used for clustering.
    """
    fig, axes = plt.subplots(2, 2, figsize=fig_size)  # Create 2x2 subplots

    silhouette_avgs = []
    for idx, n_clusters in enumerate(range_n_clusters):
        ax = axes[idx // 2, idx % 2]  # Access subplot by index
        silhouette_avg = plot_silhouette(
            ax, n_clusters, selected_data
        )  # Plot silhouette analysis
        silhouette_avgs.append(silhouette_avg)

    plt.tight_layout()  # Adjust subplot layout
    plt.savefig("./data/data_photo/plot_silhouette.pdf")  # Save figure
    plt.show()  # Show plot

    print(f"Average silhouette scores: {silhouette_avgs}")
    return silhouette_avgs


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the Dunn index for the given cluster labels.

    Args:
        X (np.ndarray): Data points.
        labels (np.ndarray): Cluster labels.

    Returns:
        float: Dunn index value.
    """
    distances = pairwise_distances(X)  # Calculate pairwise distances
    unique_clusters = np.unique(labels)  # Get unique clusters

    min_intercluster = np.inf  # Initialize minimum intercluster distance
    max_intracluster = 0  # Initialize maximum intracluster distance

    for i in unique_clusters:
        for j in unique_clusters:
            if i != j:
                intercluster_distances = distances[labels == i, :][
                    :, labels == j
                ]  # Calculate intercluster distances
                if intercluster_distances.size > 0:
                    min_intercluster = min(
                        min_intercluster, np.min(intercluster_distances)
                    )  # Update minimum intercluster distance

        intracluster_distances = distances[labels == i, :][
            :, labels == i
        ]  # Calculate intracluster distances
        if intracluster_distances.size > 0:
            max_intracluster = max(
                max_intracluster, np.max(intracluster_distances)
            )  # Update maximum intracluster distance

    if max_intracluster == 0:
        return 0  # To avoid division by zero

    return min_intercluster / max_intracluster  # Return Dunn index


def evaluate_clusters_and_plot(X: pd.DataFrame, max_clusters: int):
    """
    Evaluates clustering for a range of cluster numbers and plots Dunn Index.

    Args:
        X (pd.DataFrame): Data points.
        max_clusters (int): Maximum number of clusters to evaluate.

    Returns:
        list: Dunn index values for each number of clusters.
    """
    dunn_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)  # Initialize KMeans
        labels = kmeans.fit_predict(X)  # Predict cluster labels
        dunn_score = dunn_index(X, labels)  # Calculate Dunn index
        dunn_scores.append(dunn_score)  # Append Dunn index to list
        print(f"Dunn Index for k={k}: {dunn_score}")

    plt.figure(figsize=fig_size)  # Create figure
    plt.plot(
        range(2, max_clusters + 1),
        dunn_scores,
        "o-",
        color="red",
        linewidth=4,
        markersize=12,
        markeredgewidth=4,
        markerfacecolor="pink",
        markeredgecolor="darkgreen",
        label="Dunn Index per Cluster",
    )  # Plot Dunn index
    plt.xlabel("Number of Clusters")  # Set x-label
    plt.ylabel("Dunn Index")  # Set y-label
    plt.title("Dunn Index for Various Numbers of Clusters")  # Set title
    plt.legend(
        title="Optimization of Smart-Villages Clusters via Dunn Index",
        title_fontsize=14,
        prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
        loc="upper right",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="darkgrey",
        edgecolor="yellow",
    )  # Add legend
    mplcyberpunk.add_glow_effects()  # Add glow effects
    plt.grid(True)  # Show grid
    plt.show()  # Show plot

    return dunn_scores


def create_pipelines(n_clusters=2) -> tuple:
    """
    Create preprocessing and clustering pipelines.

    Args:
        n_clusters (int): Number of clusters to use in KMeans.

    Returns:
        tuple: Returns preprocessing, clustering, and combined pipelines.
    """
    preprocessor = Pipeline(
        [
            ("scaler", MinMaxScaler()),  # Add MinMaxScaler to pipeline
            ("pca", PCA(n_components=2, random_state=42)),  # Add PCA to pipeline
        ]
    )

    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=100,
                    max_iter=1000,
                    random_state=42,
                ),
            ),  # Add KMeans to pipeline
        ]
    )

    combined_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),  # Add preprocessor to pipeline
            ("clusterer", clusterer),  # Add clusterer to pipeline
        ]
    )

    return preprocessor, clusterer, combined_pipeline  # Return pipelines


def fit_pipeline(pipeline: Pipeline, data: pd.DataFrame) -> tuple:
    """
    Fit the pipeline to the data and extract results.

    Args:
        pipeline (Pipeline): The combined preprocessing and clustering pipeline.
        data (pd.DataFrame): The data to cluster.

    Returns:
        tuple: DataFrame with principal components, cluster assignments, and centroids.
    """
    pipeline.fit(data)  # Fit pipeline
    preprocessed_data = pipeline["preprocessor"].transform(
        data
    )  # Transform data using preprocessor
    predicted_labels = pipeline["clusterer"]["kmeans"].labels_  # Get predicted labels
    centroids = pipeline["clusterer"]["kmeans"].cluster_centers_  # Get cluster centers
    silhouette_avg = silhouette_score(
        preprocessed_data, predicted_labels
    )  # Calculate silhouette score

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Centroids of the clusters: {centroids}")

    pcadf = pd.DataFrame(
        preprocessed_data, columns=["component_1", "component_2"]
    )  # Create DataFrame for PCA components
    pcadf["predicted_cluster"] = predicted_labels  # Add predicted clusters to DataFrame

    return pcadf, centroids  # Return DataFrame and centroids


def display_cluster_scatter_plot(data: pd.DataFrame, centroids: np.ndarray):
    """
    Plot the clustered data.

    Args:
        data (pd.DataFrame): The DataFrame containing the principal components and cluster assignments.
        centroids (np.ndarray): Cluster centroids.
    """
    plt.figure(figsize=fig_size)  # Create figure
    scatter = sns.scatterplot(
        x="component_1",
        y="component_2",
        s=50,
        data=data,
        hue="predicted_cluster",
        palette="Set2",
        legend="full",
    )  # Create scatter plot
    plt.scatter(
        centroids[:, 0], centroids[:, 1], s=200, c="red", label="Centroids", marker="X"
    )  # Plot centroids
    scatter.set_title("Results of clustering of Smart-Villages data")  # Set title
    legend = plt.legend(
        loc="upper right",
        borderaxespad=0.0,
        title="Cluster IDs",
        title_fontsize=14,
        prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
        labelcolor="white",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="darkgrey",
        edgecolor="red",
    )  # Add legend
    plt.setp(legend.get_title(), color="white")  # Set legend title color
    plt.show()  # Show plot


def plot_explained_variance(data: pd.DataFrame):
    """
    Plot the cumulative explained variance by the PCA.

    Args:
        data (pd.DataFrame): The original dataset.
    """
    pca = PCA().fit(data)  # Fit PCA
    explained_variance = pca.explained_variance_ratio_  # Get explained variance ratio
    cumulative_variance = np.cumsum(
        explained_variance
    )  # Calculate cumulative explained variance

    plt.figure(figsize=(16.8, 14.4))  # Create figure
    (line,) = plt.plot(
        cumulative_variance, label="Cumulative Explained Variance", color="green"
    )  # Plot cumulative explained variance
    plt.xlabel("Number of Components")  # Set x-label
    plt.ylabel("Cumulative Explained Variance")  # Set y-label
    legend = plt.legend(
        handles=[line],
        loc="best",
        prop={"family": "Times New Roman", "size": 14, "weight": "bold"},
        labelcolor="white",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="darkgrey",
        edgecolor="red",
    )  # Add legend
    plt.setp(legend.get_title(), color="white")  # Set legend title color
    mplcyberpunk.add_glow_effects()  # Add glow effects
    plt.show()  # Show plot


def plot_3d_clusters(X: np.ndarray, n_clusters: int, selected_columns: list):
    """
    Plots 3D scatter plots of clustered data with different combinations of features.

    Args:
        X (np.ndarray): The feature data as a numpy array.
        n_clusters (int): Number of clusters to use in KMeans.
        selected_columns (list): List of column names used for labeling the axes.
    """
    kmeans = KMeans(n_clusters=n_clusters)  # Initialize KMeans
    kmeans.fit(X)  # Fit KMeans
    labels = kmeans.labels_  # Get cluster labels

    combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
    ]  # Define feature combinations for subplots
    fig = plt.figure(figsize=(19.2, 16.8))  # Create figure

    for i, (x, y, z) in enumerate(combinations, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")  # Create 3D subplot
        sc = ax.scatter(
            X[:, x], X[:, y], X[:, z], c=labels.astype(float), cmap="jet", edgecolor="k"
        )  # Plot data points
        ax.set_xlabel(selected_columns[x])  # Set x-label
        ax.set_ylabel(selected_columns[y])  # Set y-label
        ax.set_zlabel(selected_columns[z])  # Set z-label
        ax.set_title(
            f"Clusters {n_clusters} using {selected_columns[x]}, {selected_columns[y]}, {selected_columns[z]}"
        )  # Set title

    colors = [sc.cmap(sc.norm(i)) for i in range(n_clusters)]  # Get colors for clusters
    legend_labels = [
        f"Cluster {i + 1}" for i in range(n_clusters)
    ]  # Get labels for clusters
    patches = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=legend_labels[i],
            markerfacecolor=colors[i],
            markersize=10,
        )
        for i in range(n_clusters)
    ]  # Create legend patches
    legend = plt.legend(
        handles=patches,
        bbox_to_anchor=(0.5, 0.5),
        loc="best",
        prop={"family": "Times New Roman", "size": 12, "weight": "bold"},
        labelcolor="white",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="darkgrey",
        edgecolor="red",
    )  # Add legend
    plt.setp(legend.get_title(), color="white")  # Set legend title color
    cbar = fig.colorbar(
        sc, ax=fig.axes, orientation="horizontal", shrink=0.8, pad=0.15
    )  # Add colorbar
    cbar.set_label("Cluster Label")  # Set colorbar label
    plt.show()  # Show plot


def apply_pca(scaled_data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Apply PCA to reduce the dimensionality of pre-scaled data.

    Args:
        scaled_data (np.ndarray): Pre-scaled data matrix.
        n_components (int): Number of principal components to return.

    Returns:
        np.ndarray: Data transformed into principal components.
    """
    pca = PCA(n_components=n_components)  # Initialize PCA
    principal_components = pca.fit_transform(scaled_data)  # Transform data using PCA
    return principal_components  # Return principal components


def display_cluster_umap(
    data: np.ndarray,
    predicted_clusters: np.ndarray,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
):
    """
    Apply UMAP dimensionality reduction on high-dimensional data and plot the results.

    Args:
        data (np.ndarray): High-dimensional data to be reduced.
        predicted_clusters (np.ndarray): Cluster labels for each point in data.
        n_neighbors (int): The size of local neighborhood used for manifold approximation.
        min_dist (float): The minimum distance apart that points are allowed to be in the low-dimensional representation.
        n_components (int): The number of dimensions to reduce to.

    Displays:
        Scatter plot of the data reduced to two dimensions.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    )  # Initialize UMAP
    embedding = reducer.fit_transform(data)  # Transform data using UMAP

    plt.figure(figsize=fig_size)  # Create figure
    scatter = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=predicted_clusters,
        palette=sns.color_palette("hsv", len(set(predicted_clusters))),
        s=50,
        alpha=0.6,
        edgecolor="w",
    )  # Create scatter plot
    scatter.set_title("UMAP Projection of the Clustered Data")  # Set title
    legend = plt.legend(
        loc="best",
        title="Cluster ID",
        prop={"family": "Times New Roman", "size": 12, "weight": "bold"},
        labelcolor="white",
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="darkgrey",
        edgecolor="red",
    )  # Add legend
    plt.setp(legend.get_title(), color="white")  # Set legend title color
    plt.show()  # Show plot
