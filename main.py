# main.py
from src.KMethod_cluster.k_method_clusterisation import (
    load_data,
    plot_elbow_curve,
    calculate_silhouette_coefficients,
    plot_silhouette_coefficients,
    perform_clustering,
    plot_3d_clusters,
    perform_clustering_ing, plot_multiple_silhouettes,
)

from src.fuzzy_clusterisation.fuzzy_c_means import (
    preprocess_data,
    perform_fcm_clustering,
    perform_multiple_fcm_clusterings,
    plot_multiple_clusters,
    plot_pairplot,
    plot_data_distribution,
)
from src.io.output_table import display_village_clusters

from src.pca_method.pca_cluster import (create_pipelines,
                                        fit_pipeline,
                                        plot_explained_variance,
                                        display_cluster_scatter_plot)

from src.calc_dunn.calc_dunn_index import evaluate_clusters_and_plot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
from src.umap_method.umap_cluster import display_cluster_umap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)  # Configure logging


def main():
    """
        Main function to load data, perform clustering, and visualize results.
    """
    file_path = r"src/data/VILLAGE_ok_Dist2_coord-.xlsx"
    selected_columns = [
        "RD_m1_NEAR",
        "RD_m2_NEAR",
        "RD_m3_NEAR",
        "CITY2_NEAR",
        "Kiyv_NEAR_",
        "TOWN_NEAR_",
        "Water_NEAR",
        "occup_NEAR",
        "power_NEAR",
    ]
    df, selected_data = load_data(file_path, selected_columns + ["admin4Na_1"])  # Load data

    kmeans = KMeans(n_clusters=9, random_state=42)  # Initialize KMeans
    cluster_labels = kmeans.fit_predict(
        selected_data.drop(["admin4Na_1"], axis=1)
    )  # Predict clusters
    print(f'Cluster labels: {cluster_labels}')

    # Добавить столбец с метками кластеров в DataFrame
    df['cluster_label'] = cluster_labels
    df.to_excel(file_path, index=False)

    display_village_clusters(
        selected_data, cluster_labels
    )  # Display village clusters

    logging.info(f"Obtained data:\n {df.head()}")  # Log data
    logging.info(
        f"Retrieved data from selected columns:\n {df[selected_columns].head()}"
    )  # Log selected data
    logging.info(f"Descriptive statistics:\n {selected_data.describe()}")

    n_clusters = 9
    scaled_features = df[selected_columns]  # Select features
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 100,
        "max_iter": 1000,
        "random_state": 42,
    }  # Set KMeans parameters
    scaler = StandardScaler()  # Initialize StandardScaler
    scaled_features_rate = scaler.fit_transform(scaled_features)  # Scale features

    plot_elbow_curve(
        scaled_features, kmeans_kwargs, n_clusters, scaled_features_rate
    )  # Plot elbow curve
    silhouette_coefficients, optimal_k = calculate_silhouette_coefficients(
        scaled_features, kmeans_kwargs, n_clusters
    )  # Calculate silhouette coefficients
    plot_silhouette_coefficients(
        silhouette_coefficients, optimal_k, n_clusters
    )  # Plot silhouette coefficients

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
    markers = ["o", "s", "^", "p", "*", "d", "v", "<", ">"]
    perform_clustering(
        range_n_clusters, selected_data.drop(["admin4Na_1"], axis=1), markers
    )  # Perform clustering
    perform_clustering_ing(
        [3, 4, 6, 9], selected_data
    )  # Perform clustering for specific clusters

    preprocessor, clusterer, pipe = create_pipelines(
        n_clusters=n_clusters
    )  # Create pipelines
    clustered_data, centroids = fit_pipeline(pipe, selected_data)  # Fit pipeline
    display_cluster_scatter_plot(
        clustered_data, centroids
    )  # Display cluster scatter plot
    plot_explained_variance(selected_data)  # Plot explained variance

    plot_3d_clusters(
        selected_data.values, n_clusters, selected_columns
    )  # Plot 3D clusters
    evaluate_clusters_and_plot(
        selected_data, n_clusters
    )  # Evaluate clusters and plot

    X = preprocess_data(selected_data)  # Preprocess data
    fcm, labels = perform_fcm_clustering(X, n_clusters)  # Perform FCM clustering

    models = perform_multiple_fcm_clusterings(
        X, range_n_clusters
    )  # Perform multiple FCM clusterings
    plot_multiple_clusters(X, models, range_n_clusters)  # Plot multiple clusters

    plot_data_distribution(
        selected_data
    )  # The results display that there are no severe extreme values, or outliers.

    results = selected_data.copy()  # Copy selected data
    results["Labels"] = labels  # Add labels to results
    plot_pairplot(results)  # Plot pairplot

    display_cluster_umap(selected_data.values, cluster_labels)

    plot_multiple_silhouettes(range_n_clusters, selected_data)


if __name__ == "__main__":
    main()
