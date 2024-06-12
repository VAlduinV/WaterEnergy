# main.py
import numpy as np

from src.KMethod_cluster.k_method_clusterisation import (
    load_data,
    plot_elbow_curve,
    calculate_silhouette_coefficients,
    plot_silhouette_coefficients,
    perform_clustering,
    plot_3d_clusters,
    perform_clustering_ing,
    plot_multiple_silhouettes,
)
from src.cluster_map.vis_map import MapPlotter

from src.fuzzy_clusterisation.fuzzy_c_means import (
    preprocess_data,
    perform_fcm_clustering,
    perform_multiple_fcm_clusterings,
    plot_multiple_clusters,
    plot_pairplot,
    plot_data_distribution,
)

from src.io.output_table import display_village_clusters

from src.pca_method.pca_cluster import (
    create_pipelines,
    fit_pipeline,
    plot_explained_variance,
    display_cluster_scatter_plot,
)

from src.calc_dunn.calc_dunn_index import evaluate_clusters_and_plot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
from src.umap_method.umap_cluster import display_cluster_umap

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)  # Configure logging


def run_kmeans_clustering(selected_data, selected_columns, n_clusters=4):
    numeric_data = selected_data[selected_columns].select_dtypes(include=[float, int])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(numeric_data)
    print(f"Cluster labels: {cluster_labels}")

    scaler = StandardScaler()
    scaled_features_rate = scaler.fit_transform(numeric_data)

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 100,
        "max_iter": 1000,
        "random_state": 42,
    }

    # plot_elbow_curve(numeric_data, kmeans_kwargs, n_clusters, scaled_features_rate)
    # silhouette_coefficients, optimal_k = calculate_silhouette_coefficients(
    #    numeric_data, kmeans_kwargs, n_clusters
    # )
    # plot_silhouette_coefficients(silhouette_coefficients, optimal_k, n_clusters)

    # range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
    # markers = ["o", "s", "^", "p", "*", "d", "v", "<", ">"]
    # perform_clustering(range_n_clusters, numeric_data, markers)
    # perform_clustering_ing([3, 4, 6, 9], numeric_data)

    plot_3d_clusters(numeric_data.values, n_clusters, selected_columns)
    # evaluate_clusters_and_plot(numeric_data, n_clusters)
    # display_cluster_umap(numeric_data.values, cluster_labels)
    # plot_multiple_silhouettes(range_n_clusters, numeric_data)
    return cluster_labels


def run_pca_clustering(selected_data, selected_columns, n_clusters=4):
    numeric_data = selected_data[selected_columns].select_dtypes(include=[float, int])
    preprocessor, clusterer, pipe = create_pipelines(n_clusters=n_clusters)
    clustered_data, centroids = fit_pipeline(pipe, numeric_data)
    display_cluster_scatter_plot(clustered_data, centroids)
    plot_explained_variance(numeric_data)
    print("Unique cluster labels:", clustered_data["Labels"].unique())
    return clustered_data["Labels"]


def run_fuzzy_clustering(selected_data, n_clusters=4):
    numeric_data = selected_data.select_dtypes(include=[float, int])
    X = preprocess_data(numeric_data)
    fcm, labels = perform_fcm_clustering(X, n_clusters)

    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
    models = perform_multiple_fcm_clusterings(X, range_n_clusters)
    plot_multiple_clusters(X, models, range_n_clusters)

    plot_data_distribution(numeric_data)
    results = numeric_data.copy()
    results["Labels"] = labels
    plot_pairplot(results)
    return labels


def main():
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
    df, selected_data = load_data(file_path, selected_columns + ["admin4Na_1"])

    # Choose selected_data
    kmeans_labels = run_kmeans_clustering(selected_data, selected_columns)
    df["kmeans_cluster_label"] = kmeans_labels
    output_k_means = display_village_clusters(selected_data, kmeans_labels)
    logging.info(f'Output algorithm K-means:\n {output_k_means}\n')

    # run_pca_clustering(selected_data)

    fuzzy_labels = run_fuzzy_clustering(selected_data)
    df["fuzzy_cluster_label"] = fuzzy_labels
    output_c_means = display_village_clusters(selected_data, fuzzy_labels)
    logging.info(f'Output algorithm C-means:\n {output_c_means}\n')

    pca_labels = run_pca_clustering(selected_data, selected_columns)
    df["pca_cluster_label"] = pca_labels
    output_pca_means = display_village_clusters(selected_data, pca_labels)
    logging.info(f'Output algorithm PCA:\n {output_pca_means}\n')

    df.to_excel(file_path, index=False)

    logging.info(f"Obtained data:\n {df.head()}")
    logging.info(
        f"Retrieved data from selected columns:\n {df[selected_columns].head()}"
    )
    logging.info(f"Descriptive statistics:\n {selected_data.describe()}")

    # Plotting and saving clusters on the map
    map_shapefile = r'C:\Users\prime\PycharmProjects\WaterEnergy\src\data\map_data\gadm41_UKR_1.shp'
    plotter = MapPlotter(map_shapefile)

    # Optionally filter regions
    df = plotter.filter_regions(df, include_all=False, region_col='admin4Na_1')

    pca_gdf = plotter.create_village_gdf(df, 'xcoord', 'ycoord', 'pca_cluster_label')
    plotter.plot_clustered_villages(pca_gdf, 'pca_cluster_label', 'pca_cluster_label.png', 'PCA Clustering')

    kmeans_gdf = plotter.create_village_gdf(df, 'xcoord', 'ycoord', 'kmeans_cluster_label')
    plotter.plot_clustered_villages(kmeans_gdf, 'kmeans_cluster_label', 'kmeans_clusters.png', 'K-means Clustering')
    # plotter.save_clusters_to_shapefile(kmeans_gdf, 'kmeans_cluster_label', 'kmeans_clusters.shp')

    fuzzy_gdf = plotter.create_village_gdf(df, 'xcoord', 'ycoord', 'fuzzy_cluster_label')
    plotter.plot_clustered_villages(fuzzy_gdf, 'fuzzy_cluster_label', 'fuzzy_clusters.png', 'C-means Clustering')
    # plotter.save_clusters_to_shapefile(fuzzy_gdf, 'fuzzy_cluster_label', 'fuzzy_clusters.shp')


if __name__ == "__main__":
    main()
