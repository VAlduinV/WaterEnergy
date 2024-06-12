import seaborn as sns
from fcmeans import FCM
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from src.CONSTANT.constant import *


def preprocess_data(selected_data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess the data by removing duplicates, filling missing values, and scaling.

    Args:
        selected_data (pd.DataFrame): The data to preprocess.

    Returns:
        np.ndarray: Scaled data.
    """
    selected_data = selected_data.copy()  # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    selected_data.drop_duplicates(inplace=True)  # Remove duplicates
    selected_data.fillna(selected_data.mean(), inplace=True)  # Fill missing values with column mean
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
    labels = fcm.predict(X) + 1  # Predict cluster labels and start from 1
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
        fcm = FCM(n_clusters=n_clusters, max_iter=1000, random_state=42)  # Initialize FCM
        fcm.fit(X)  # Fit FCM
        models.append(fcm)  # Append model to list
    return models


def plot_data_distribution(selected_data: pd.DataFrame):
    """
    Plot the distribution of the data.

    Args:
        selected_data (pd.DataFrame): The data to plot.
    """
    ax = selected_data.plot.box(figsize=FIG_SIZE)  # Create box plot
    ax.set_ylabel("Distance (in metres)")  # Set y-axis label using the axis object
    plt.show()  # Show plot


def plot_multiple_clusters(X: np.ndarray, models: list, n_clusters_list: list, cmap_name='rainbow'):
    """
    Plot the fuzzy clusters for multiple numbers of clusters.

    Args:
        X (np.ndarray): The data to plot.
        models (list): List of FCM models.
        n_clusters_list (list): List of numbers of clusters.
        cmap_name (str): Name of the colormap to use for the clusters.
    """
    num_clusters = len(n_clusters_list)
    rows = int(np.ceil(np.sqrt(num_clusters)))  # Number of rows in the subplot grid
    cols = int(np.ceil(num_clusters / rows))  # Number of columns in the subplot grid
    f, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Create subplots

    cmap = cm.get_cmap(cmap_name)  # Get the colormap

    for n_clusters, model, axe in zip(n_clusters_list, models, axes.ravel()):
        pc = model.partition_coefficient  # Partition coefficient
        pec = model.partition_entropy_coefficient  # Partition entropy coefficient
        fcm_centers = model.centers  # FCM centers
        fcm_labels = model.predict(X) + 1  # Predict labels and start from 1

        axe.scatter(X[:, 0], X[:, 1], c=fcm_labels, cmap=cmap, alpha=0.9)  # Plot data points
        axe.scatter(fcm_centers[:, 0], fcm_centers[:, 1], marker="+", s=400, c="#0000ff")  # Plot centers
        axe.set_title(f"n_clusters = {n_clusters}, PC = {pc:.3f}, PEC = {pec:.3f}")  # Set title
        # axe.set_xticks([])  # Remove x-ticks
        # axe.set_yticks([])  # Remove y-ticks

    for j in range(len(n_clusters_list), len(axes.flatten())):
        f.delaxes(axes.flatten()[j])  # Remove extra subplots

    # Adding legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / num_clusters), markersize=10,
                          label=f'Cluster {i + 1}') for i in range(max(n_clusters_list))]
    f.legend(handles=handles, loc='center right', title='Clusters')
    plt.tight_layout()  # Adjust layout
    plt.show()  # Show plot


def plot_pairplot(results: pd.DataFrame):
    """
    Plot the pairplot of the results.

    Args:
        results (pd.DataFrame): The data to plot.
    """
    sns.pairplot(results, hue="Labels", palette="Dark2")  # Create pairplot
    plt.show()  # Show plot
