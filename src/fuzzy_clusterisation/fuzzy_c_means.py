# WaterEnergy/fuzzy_clusterisation/fuzzy_c_means.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fcmeans import FCM
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(selected_data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess the data by removing duplicates, filling missing values, and scaling.

    Args:
        selected_data (pd.DataFrame): The data to preprocess.

    Returns:
        np.ndarray: Scaled data.
    """
    selected_data = (
        selected_data.copy()
    )  # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    selected_data.drop_duplicates(inplace=True)  # Remove duplicates
    selected_data.fillna(
        selected_data.mean(), inplace=True
    )  # Fill missing values with column mean
    scaler = StandardScaler()  # Initialize the StandardScaler
    X = scaler.fit_transform(selected_data)  # Scale the data
    return X


def perform_fcm_clustering(X: np.ndarray, n_clusters: int) -> tuple:
    """
    Perform FCM clustering on the data.

    Args:
        X (np.ndarray): The data to cluster.
        n_clusters (int): Number of clusters.

    Returns:
        tuple: FCM model and cluster labels.
    """
    fcm = FCM(n_clusters=n_clusters, max_iter=1000, random_state=42)  # Initialize FCM
    fcm.fit(X)  # Fit FCM
    labels = fcm.predict(X)  # Predict cluster labels
    return fcm, labels


def perform_multiple_fcm_clusterings(X: np.ndarray, n_clusters_list: list) -> list:
    """
    Perform FCM clustering for multiple numbers of clusters.

    Args:
        X (np.ndarray): The data to cluster.
        n_clusters_list (list): List of numbers of clusters to try.

    Returns:
        list: List of FCM models.
    """
    models = []
    for n_clusters in n_clusters_list:
        fcm = FCM(n_clusters=n_clusters)  # Initialize FCM
        fcm.fit(X)  # Fit FCM
        models.append(fcm)  # Append model to list
    return models


def plot_data_distribution(selected_data: pd.DataFrame):
    """
    Plot the distribution of the data.

    Args:
        selected_data (pd.DataFrame): The data to plot.
    """
    plt.style.use("ggplot")  # Use ggplot style
    selected_data.plot.box(figsize=(19.2, 16.8))  # Create box plot
    plt.show()  # Show plot


def plot_clusters_fuzzy(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
    """
    Plot the fuzzy clusters.

    Args:
        X (np.ndarray): The data to plot.
        labels (np.ndarray): Cluster labels.
        centroids (np.ndarray): Cluster centroids.
    """
    colors = ["black", "green", "red", "c", "m", "yellow", "pink", "violet", "blue"]
    n_clusters = len(np.unique(labels))  # Number of clusters
    plt.figure(figsize=(19.2, 16.8))  # Create figure
    for i in range(n_clusters):
        plt.scatter(
            X[labels == i, 0], X[labels == i, 1], c=colors[i]
        )  # Plot data points
    plt.scatter(
        centroids[:, 0], centroids[:, 1], marker="x", s=100, c="#050505"
    )  # Plot centroids
    plt.xlabel("RD_m1_NEAR")  # Set x-label
    plt.ylabel("RD_m2_NEAR")  # Set y-label
    plt.show()  # Show plot


def plot_multiple_clusters(X: np.ndarray, models: list, n_clusters_list: list):
    """
    Plot the fuzzy clusters for multiple numbers of clusters.

    Args:
        X (np.ndarray): The data to plot.
        models (list): List of FCM models.
        n_clusters_list (list): List of numbers of clusters.
    """
    num_clusters = len(n_clusters_list)
    rows = int(np.ceil(np.sqrt(num_clusters)))  # Number of rows in the subplot grid
    cols = int(np.ceil(num_clusters / rows))  # Number of columns in the subplot grid
    f, axes = plt.subplots(rows, cols, figsize=(19.2, 16.8))  # Create subplots
    for n_clusters, model, axe in zip(n_clusters_list, models, axes.ravel()):
        pc = model.partition_coefficient  # Partition coefficient
        pec = model.partition_entropy_coefficient  # Partition entropy coefficient
        fcm_centers = model.centers  # FCM centers
        fcm_labels = model.predict(X)  # Predict labels
        axe.scatter(X[:, 0], X[:, 1], c=fcm_labels, alpha=0.9)  # Plot data points
        axe.scatter(
            fcm_centers[:, 0], fcm_centers[:, 1], marker="+", s=200, c="r"
        )  # Plot centers
        axe.set_title(
            f"n_clusters = {n_clusters}, PC = {pc:.3f}, PEC = {pec:.3f}"
        )  # Set title
    plt.show()  # Show plot


def plot_pairplot(results: pd.DataFrame):
    """
    Plot the pairplot of the results.

    Args:
        results (pd.DataFrame): The data to plot.
    """
    sns.pairplot(results, hue="Labels", palette="Dark2")  # Create pairplot
    plt.show()  # Show plot
