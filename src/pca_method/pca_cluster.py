from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import numpy as np
from src.CONSTANT.constant import *
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, required for 3D plotting


def create_pipelines(n_clusters=9) -> tuple:
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
    plt.figure(figsize=FIG_SIZE)  # Create figure
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

    plt.figure(figsize=FIG_SIZE)  # Create figure
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
