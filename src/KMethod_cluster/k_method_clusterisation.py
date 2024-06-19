# WaterEnergy/KMethod_cluster/k_method_clusterisation.py
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, required for 3D plotting
from src.CONSTANT.constant import *


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


def plot_elbow_curve(features: pd.DataFrame, kmeans_kwargs: dict, n_clusters: int, scaled_features: np.ndarray) -> int:
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

    plt.figure(figsize=FIG_SIZE)  # Create figure
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
    plt.savefig("./src/data/data_photo/elbow_curve.pdf")  # Save figure
    plt.show()  # Show plot

    print(f"Optimal number of clusters according to the elbow method: {kl.elbow}\n")
    print(f"SSE for each k: {sse}\n")

    return kl.elbow


def calculate_silhouette_coefficients(scaled_features: np.ndarray, kmeans_kwargs: dict, n_clusters: int) -> tuple:
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

    print(f"Silhouette coefficients for each k: {silhouette_coefficients}\n")
    print(f"Optimal number of clusters according to silhouette method: {optimal_k}\n")

    return silhouette_coefficients, optimal_k


def plot_silhouette_coefficients(silhouette_coefficients: list, optimal_k: int, n_clusters: int):
    """
    Plots the silhouette coefficients for a range of cluster numbers.

    Args:
        silhouette_coefficients (list): List of silhouette coefficients.
        optimal_k (int): Optimal number of clusters.
        n_clusters (int): Maximum number of clusters evaluated.
    """
    plt.figure(figsize=FIG_SIZE)  # Create figure
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
    plt.savefig("./src/data/data_photo/silhouette_coefficients.pdf")  # Save figure
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

    print(f"Average silhouette score for n_clusters = {n_clusters}: {silhouette_avg}\n")

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
            -0.05, y_lower + 0.5 * size_cluster_i, str(i + 1)  # Use i + 1 to start from 1
        )  # Add text label for cluster i
        y_lower = y_upper + 10  # Update y_lower for next cluster

    ax.set_title("The silhouette plot for the various clusters.")  # Set title
    ax.set_xlabel("The silhouette coefficient values")  # Set x-label
    ax.set_ylabel("Cluster label")  # Set y-label
    ax.axvline(
        x=silhouette_avg, color="red", linestyle="--"
    )  # Add vertical line for average silhouette score

    return silhouette_avg


def perform_clustering(range_n_clusters: list, selected_data: pd.DataFrame, markers: list):
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
        plt.tight_layout()  # Adjust subplot layout
        plt.show()  # Show plot

    return silhouette_avgs


def perform_clustering_ing(range_n_clusters, selected_data):
    """
    Perform clustering analysis and visualization for a specified range of cluster counts.

    Args:
        range_n_clusters (list): List of integers representing the number of clusters to analyze.
        selected_data (pd.DataFrame): Data used for clustering.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)  # Create 2x2 subplots

    silhouette_avgs = []
    for idx, n_clusters in enumerate(range_n_clusters):
        ax = axes[idx // 2, idx % 2]  # Access subplot by index
        silhouette_avg = plot_silhouette(
            ax, n_clusters, selected_data
        )  # Plot silhouette analysis
        silhouette_avgs.append(silhouette_avg)

    plt.tight_layout()  # Adjust subplot layout
    plt.savefig("./src/data/data_photo/plot_silhouette.pdf")  # Save figure
    plt.show()  # Show plot

    print(f"Average silhouette scores: {silhouette_avgs}\n")
    return silhouette_avgs


def plot_multiple_silhouettes(range_n_clusters: list, selected_data: pd.DataFrame):
    """
    Plots multiple silhouette analyses on a single figure.

    Args:
        range_n_clusters (list): List of integers representing the number of clusters to analyze.
        selected_data (pd.DataFrame): The data on which to perform clustering.
    """
    fig, axes = plt.subplots(3, 3, figsize=FIG_SIZE)  # Создаем фигуру с сеткой 3x3
    axes = axes.flatten()  # Разворачиваем массив осей для удобства итерации

    for idx, n_clusters in enumerate(range_n_clusters):
        ax = axes[idx]
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(selected_data) + (n_clusters + 1) * 10])
        plot_silhouette(ax, n_clusters, selected_data)
        ax.set_title(f"n_clusters = {n_clusters}")

    # Удаляем лишние оси
    for j in range(len(range_n_clusters), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Настраиваем расположение подзаголовков
    plt.show()  # Показываем фигуру


def plot_3d_clusters(X: np.ndarray, n_clusters: int, selected_columns: list, translated_labels: dict):
    """
    Plots 3D scatter plots of clustered data with different combinations of features.

    Args:
        X (np.ndarray): The feature data as a numpy array.
        n_clusters (int): Number of clusters to use in KMeans.
        selected_columns (list): List of column names used for labeling the axes.
        translated_labels (dict): Dictionary mapping original column names to new labels.
    """
    kmeans = KMeans(n_clusters=n_clusters)  # Initialize KMeans
    kmeans.fit(X)  # Fit KMeans
    labels = kmeans.labels_ + 1  # Get cluster labels and start from 1

    combinations = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
    ]  # Define feature combinations for subplots
    fig = plt.figure(figsize=(20, 10))  # Create figure with larger size

    cmap = plt.cm.get_cmap('jet', n_clusters)  # Get colormap

    for i, (x, y, z) in enumerate(combinations, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")  # Create 3D subplot
        sc = ax.scatter(
            X[:, x], X[:, y], X[:, z], c=labels, cmap=cmap, edgecolor="k"
        )  # Plot data points
        ax.set_xlabel(selected_columns[x], fontsize=12)  # Set x-label
        ax.set_ylabel(selected_columns[y], fontsize=12)  # Set y-label
        ax.set_zlabel(selected_columns[z], fontsize=12)  # Set z-label
        ax.set_title(
            f"Clusters {n_clusters} using {selected_columns[x]} ({translated_labels[selected_columns[x]]}),\n"
            f"{selected_columns[y]} ({translated_labels[selected_columns[y]]}),\n"
            f"{selected_columns[z]} ({translated_labels[selected_columns[z]]})\n",
            fontsize=14
        )  # Set title

    colors = [cmap(i) for i in range(n_clusters)]  # Get colors for clusters
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
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        prop={"size": 10},
        shadow=True,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor="lightgrey",
        edgecolor="black"
    )  # Add legend
    plt.setp(legend.get_title(), fontsize=12)  # Set legend title font size
    cbar = fig.colorbar(
        sc, ax=fig.axes, orientation="horizontal", shrink=0.8, pad=0.15
    )  # Add colorbar
    cbar.set_label("Cluster Label", fontsize=12)  # Set colorbar label
    plt.show()  # Show plot
